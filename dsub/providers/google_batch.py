# Copyright 2022 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provider for running jobs on Google Cloud Platform.

This module implements job creation, listing, and canceling using the
Google Batch v1 APIs.
"""

import ast
import io
import json
import operator
import os
import re
import sys
import textwrap
from typing import Dict, List, NamedTuple, Set

from google.api_core import exceptions as google_exceptions
import googleapiclient.http

from ..lib import dsub_util
from ..lib import job_model
from ..lib import param_util
from ..lib import providers_util
from . import base
from . import google_base
from . import google_batch_operations
from . import google_custom_machine
from . import google_utils

# pylint: disable=g-import-not-at-top
try:
  from google.cloud import batch_v1
except ImportError:
  # TODO: Remove conditional import when batch library is available
  from . import batch_dummy as batch_v1
# pylint: enable=g-import-not-at-top
_PROVIDER_NAME = 'google-batch'
# Index of the prepare action in the runnable list
_PREPARE_INDEX = 1
_TASK_GROUP_NAME = 'group0'
_DEFAULT_LIST_PAGE_SIZE = 500
_MAX_TASKS_PER_JOB = 100000
_MAX_INLINE_JOB_DEFINITION_BYTES = 900000
_TASK_ENV_PREFIX = '_DSUB_BATCH_{action}_'
_TASK_MANIFEST_INDEX_RECORD_SIZE = 42
_TASK_MANIFEST_URI_VARNAME = '_DSUB_TASK_MANIFEST_URI'
_TASK_MANIFEST_INDEX_URI_VARNAME = '_DSUB_TASK_MANIFEST_INDEX_URI'
_TASK_MANIFEST_INDEX_RECORD_SIZE_VARNAME = (
    '_DSUB_TASK_MANIFEST_INDEX_RECORD_SIZE'
)
_TASK_MANIFEST_USER_PROJECT_VARNAME = '_DSUB_TASK_MANIFEST_USER_PROJECT'

# Create file provider whitelist.
_SUPPORTED_FILE_PROVIDERS = frozenset([job_model.P_GCS])
_SUPPORTED_LOGGING_PROVIDERS = _SUPPORTED_FILE_PROVIDERS
_SUPPORTED_INPUT_PROVIDERS = _SUPPORTED_FILE_PROVIDERS
_SUPPORTED_OUTPUT_PROVIDERS = _SUPPORTED_FILE_PROVIDERS

# Mount point for the data disk in the user's Docker container
_VOLUME_MOUNT_POINT = '/mnt/disks/data'
_DATA_MOUNT_POINT = '/mnt/data'
_TASK_ENV_FILE = f'{_DATA_MOUNT_POINT}/.dsub/task-environment.sh'

# These are documented (providers/README.md) as being read/write to user
# commands.
_SCRIPT_DIR = f'{_DATA_MOUNT_POINT}/script'
_TMP_DIR = f'{_DATA_MOUNT_POINT}/tmp'
_WORKING_DIR = f'{_DATA_MOUNT_POINT}/workingdir'

# These are visible to the user task; not yet documented, as we'd *like* to
# find a way to have them visible only to the logging tasks.
_BATCH_LOG_DIR = f'{_VOLUME_MOUNT_POINT}/.logging'
_LOGGING_DIR = f'{_DATA_MOUNT_POINT}/.logging'

_LOG_FILTER_VAR = '_LOG_FILTER_REPR'
_LOG_FILTER_SCRIPT_PATH = f'{_DATA_MOUNT_POINT}/.log_filter_script.py'

_TASK_MANIFEST_BOOTSTRAP_CMD = textwrap.dedent(r'''\
  set -o errexit
  set -o nounset
  set -o pipefail

  readonly task_root="/mnt/data/.dsub"
  readonly task_json="${task_root}/task-environment.json"
  readonly task_shell="${task_root}/task-environment.sh"
  readonly index_size="${_DSUB_TASK_MANIFEST_INDEX_RECORD_SIZE}"
  readonly index_start="$((BATCH_TASK_INDEX * index_size))"
  readonly index_end="$((index_start + index_size - 1))"

  user_project_args=()
  if [[ -n "${_DSUB_TASK_MANIFEST_USER_PROJECT}" ]]; then
    user_project_args=(-u "${_DSUB_TASK_MANIFEST_USER_PROJECT}")
  fi

  mkdir -p "${task_root}"
  read -r task_start task_end < <(
    gsutil "${user_project_args[@]}" -q cat \
      -r "${index_start}-${index_end}" \
      "${_DSUB_TASK_MANIFEST_INDEX_URI}"
  )
  gsutil "${user_project_args[@]}" -q cat \
    -r "${task_start}-${task_end}" \
    "${_DSUB_TASK_MANIFEST_URI}" > "${task_json}"

  python3 - "${task_json}" "${task_shell}" <<'PY'
import json
import re
import shlex
import sys

with open(sys.argv[1], encoding='utf-8') as source:
  environment = json.load(source)

with open(sys.argv[2], 'w', encoding='utf-8') as destination:
  for name, value in environment.items():
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
      raise ValueError(f'Invalid environment variable name: {name!r}')
    destination.write(f'export {name}={shlex.quote(str(value))}\n')
PY
''')


class _TaskManifest(NamedTuple):
  """A staged, random-access mapping from Batch indices to dsub tasks."""

  uri: str
  index_uri: str
  user_project: str
  data: bytes
  index: bytes
  common_environment: Dict[str, str]

# _LOG_FILTER_PYTHON is a block of Python code to execute in both the
# "continuous_logging" and "final_logging" tasks.
#
# Batch API will eventually create three log files in the _BATCH_LOG_FILE_PATH
# directory. They are named:
#
# - output-*.log
# - stdout-*.log
# - stderr-*.log
#
# We will be creating a "staging" location for each of these files.
#
# If any of the batch log files don't exist, touch the associated staging file.
#
# If the output batch log file exists, copy it directly to the staging
# location.
#
# If the stdout/stderr batch log files exist, copy them to their staging
# location. Then filter it so that only user-action logs exist and the prefixes
# are removed. The prefixes look something like:
# [batch_task_logs]<datetime> ERROR:
#   [task_id:task/<job_uid>,runnable_index:<action_number>]

# pylint: disable=anomalous-backslash-in-string
_LOG_FILTER_PYTHON = textwrap.dedent(r"""
import fileinput
import glob
import re
import shutil
import sys
from pathlib import Path

LOGGING_DIR = sys.argv[1]
LOG_FILE_PATH = sys.argv[2]
STDOUT_FILE_PATH = sys.argv[3]
STDERR_FILE_PATH = sys.argv[4]
USER_TASK = sys.argv[5]

def filter_log_file(staging_path: str, stream_string: str):
  # Replaces lines in file inplace
  for line in fileinput.input(staging_path, inplace=True):
    re_search_string = fr"^\[batch_task_logs\].*{stream_string}: \[task_id:task\/.*runnable_index:{USER_TASK}] (.*)"
    match = re.search(re_search_string, line)
    if match:
      modified_line = match.group(1)
      print(modified_line)

def copy_log_to_staging(glob_str: str, staging_path: str, filter_str: str = None):
  # Check if log files exist, and copy to their staging location
  matching_files = list(Path(LOGGING_DIR).glob(glob_str))
  if matching_files:
    assert(len(matching_files) == 1)
    shutil.copy(matching_files[0], staging_path)
    if filter_str:
      filter_log_file(staging_path, filter_str)
  else:
    Path(staging_path).touch()

# We know the log file is named output-<job-uid>.log
# and the stdout/stderr files are named stdout-<job-uid>.log and
# stderr-<job-uid>.log
copy_log_to_staging("output-*.log", LOG_FILE_PATH)
copy_log_to_staging("stdout-*.log", STDOUT_FILE_PATH, filter_str="INFO")
copy_log_to_staging("stderr-*.log", STDERR_FILE_PATH, filter_str="ERROR")
""")
# pylint: enable=anomalous-backslash-in-string

_LOG_CP = textwrap.dedent("""
  python3 "{log_filter_script_path}" \
      "${{LOGGING_DIR}}" \
      "${{LOGGING_DIR}}/log.txt" \
      "${{LOGGING_DIR}}/stdout.txt" \
      "${{LOGGING_DIR}}/stderr.txt" \
      "{user_action}"

  gsutil_cp "${{LOGGING_DIR}}/stdout.txt" "${{STDOUT_PATH}}" "text/plain" "${{USER_PROJECT}}" &
  STDOUT_PID=$!
  gsutil_cp "${{LOGGING_DIR}}/stderr.txt" "${{STDERR_PATH}}" "text/plain"  "${{USER_PROJECT}}" &
  STDERR_PID=$!
  gsutil_cp "${{LOGGING_DIR}}/log.txt" "${{LOGGING_PATH}}" "text/plain" "${{USER_PROJECT}}" &
  LOG_PID=$!

  wait "${{STDOUT_PID}}"
  wait "${{STDERR_PID}}"
  wait "${{LOG_PID}}"
""")

_FINAL_LOGGING_CMD = textwrap.dedent("""\
  set -o errexit
  set -o nounset
  set -o pipefail

  readonly LOGGING_DIR="{logging_dir}"

  # Flag the continuous logging command to stop
  touch "${{LOGGING_DIR}}/.stop_logging"

  {log_msg_fn}
  {gsutil_cp_fn}

  {log_cp}
""")

# Keep logging until the final logging action starts
_CONTINUOUS_LOGGING_CMD = textwrap.dedent("""\
  set -o errexit
  set -o nounset
  set -o pipefail

  readonly LOGGING_DIR="{logging_dir}"

  {log_msg_fn}
  {gsutil_cp_fn}

  # Make sure the logging work directory exists
  mkdir -p "${{LOGGING_DIR}}"

  # Prep the log filter script
  echo "${{{log_filter_var}}}" \
    | python3 -c '{python_decode_script}' \
    > "{log_filter_script_path}"
  chmod a+x "{log_filter_script_path}"

  while [[ ! -e "${{LOGGING_DIR}}/.stop_logging" ]]; do
    {log_cp}

    sleep "{log_interval}"
  done
""")


_EVENT_REGEX_MAP = {
    'scheduled': re.compile('^Job state is set from QUEUED to SCHEDULED'),
    'start': re.compile('^Job state is set from SCHEDULED to RUNNING'),
    'ok': re.compile('^Job state is set from RUNNING to SUCCEEDED'),
    'fail': re.compile('^Job state is set from .+? to FAILED'),
    'cancellation-in-progress': re.compile(
        '^Job state is set from .+? to CANCELLATION_IN_PROGRESS'
    ),
    'canceled': re.compile('^Job state is set from .+? to CANCELLED'),
}


class GoogleBatchEventMap(object):
  """Helper for extracing a set of normalized, filtered operation events."""

  def __init__(self, op: batch_v1.types.Job):
    self._op = op

  def get_filtered_normalized_events(self):
    """Map and filter the batch API events down to events of interest.

    Returns:
      A list of maps containing the normalized, filtered events.
    """
    events = {}
    for event in google_batch_operations.get_status_events(self._op):
      mapped, _ = self._map(event)
      name = mapped['name']

      events[name] = mapped

    return sorted(list(events.values()), key=operator.itemgetter('event-time'))

  def _map(self, event):
    """Extract elements from a Batch status event and map to a named event."""
    description = event.description
    event_time = event.event_time.rfc3339()

    for name, regex in _EVENT_REGEX_MAP.items():
      match = regex.match(description)
      if match:
        return {'name': name, 'event-time': event_time}, match

    return {'name': description, 'event-time': event_time}, None


class GoogleBatchOperation(base.Task):
  """dsub task wrapper around a Batch API Job and one of its Tasks."""

  def __init__(
      self,
      operation_data: batch_v1.types.Job,
      task_data: batch_v1.types.Task = None,
      task_environment: Dict[str, str] = None,
      task_index: int = 0,
  ):
    self._op = operation_data
    self._task = task_data
    self._task_index = (
        self._get_task_index(task_data) if task_data is not None else task_index
    )
    self._manifest_task_environment = task_environment
    self._job_descriptor = self._try_op_to_job_descriptor()

  def raw_task_data(self):
    return self._task or self._op

  @staticmethod
  def _get_task_index(task):
    if not task:
      return 0
    match = re.search(r'(\d+)$', task.name)
    if not match:
      raise ValueError(f'Unable to determine Batch task index: {task.name}')
    return int(match.group(1))

  def _get_task_environment(self):
    if self._manifest_task_environment is not None:
      return self._manifest_task_environment
    environment = google_batch_operations.get_task_environment(
        self._op, self._task_index
    )
    if environment:
      return environment

    # Single-task Jobs store their environment once on TaskSpec. A manifest
    # Job is reconstructed by the provider and passed above; do not mistake
    # its shared environment for a complete task record.
    environment = google_batch_operations.get_task_spec_environment(self._op)
    if environment.get(_TASK_MANIFEST_URI_VARNAME):
      return {}
    return environment

  @staticmethod
  def _task_environment_name(action, name):
    return _TASK_ENV_PREFIX.format(action=action.upper()) + name

  def _try_op_to_job_descriptor(self):
    # New multi-task Jobs store each task's descriptor in its TaskGroup
    # environment. Fall back to the runnable environment for Jobs submitted by
    # older dsub versions, where each Google Batch Job represented one task.
    env = self._get_task_environment()
    meta = env.get(
        self._task_environment_name(
            'prepare', google_utils.META_YAML_VARNAME
        )
    )
    if not meta:
      env = google_batch_operations.get_environment(self._op, _PREPARE_INDEX)
      meta = env.get(google_utils.META_YAML_VARNAME) if env else None
    if not meta:
      return None

    return job_model.JobDescriptor.from_yaml(ast.literal_eval(meta))

  def get_field(self, field: str, default: str = None):
    """Returns a value from the operation for a specific set of field names.

    This is the implementation of base.Task's abstract get_field method. See
    base.py get_field for more details.

    Args:
      field: a dsub-specific job metadata key
      default: default value to return if field does not exist or is empty.

    Returns:
      A text string for the field or a list for 'inputs'.

    Raises:
      ValueError: if the field label is not supported by the operation
    """
    value = None
    if field == 'internal-id':
      value = self._op.name
    elif field == 'user-project':
      if self._job_descriptor:
        value = self._job_descriptor.job_metadata.get(field)
    elif field in ['task-id', 'task-attempt']:
      if self._job_descriptor:
        value = self._job_descriptor.task_descriptors[0].task_metadata.get(
            field
        )
      if value is None:
        value = google_batch_operations.get_label(self._op, field)
    elif field in ['job-id', 'job-name', 'user-id', 'dsub-version']:
      if self._job_descriptor:
        value = self._job_descriptor.job_metadata.get(field)
      if value is None:
        value = google_batch_operations.get_label(self._op, field)
    elif field == 'task-status':
      value = self._operation_status()
    elif field == 'logging':
      if self._job_descriptor:
        # The job_resources will contain the "--logging" value.
        # The task_resources will contain the resolved logging path.
        # Return the resolved logging path.
        task_resources = self._job_descriptor.task_descriptors[0].task_resources
        value = task_resources.logging_path
    elif field in ['envs', 'labels']:
      if self._job_descriptor:
        items = providers_util.get_job_and_task_param(
            self._job_descriptor.job_params,
            self._job_descriptor.task_descriptors[0].task_params,
            field,
        )
        value = {item.name: item.value for item in items}
    elif field in [
        'inputs',
        'outputs',
        'input-recursives',
        'output-recursives',
    ]:
      if self._job_descriptor:
        value = {}
        items = providers_util.get_job_and_task_param(
            self._job_descriptor.job_params,
            self._job_descriptor.task_descriptors[0].task_params,
            field,
        )
        value.update({item.name: item.value for item in items})
    elif field == 'mounts':
      if self._job_descriptor:
        items = providers_util.get_job_and_task_param(
            self._job_descriptor.job_params,
            self._job_descriptor.task_descriptors[0].task_params,
            field,
        )
        value = {item.name: item.value for item in items}
    elif field == 'provider':
      return _PROVIDER_NAME
    elif field == 'provider-attributes':
      value = {
          'boot-disk-size': google_batch_operations.get_boot_disk_size(
              self._op
          ),
          'disk-size': google_batch_operations.get_disk_size(self._op),
          'disk-type': google_batch_operations.get_disk_type(self._op),
          'machine-type': google_batch_operations.get_machine_type(self._op),
          'regions': google_batch_operations.get_regions(self._op),
          'zones': google_batch_operations.get_zones(self._op),
          'preemptible': google_batch_operations.get_preemptible(self._op),
      }
    elif field == 'events':
      value = GoogleBatchEventMap(
          self._task or self._op
      ).get_filtered_normalized_events()
    elif field == 'script-name':
      if self._job_descriptor:
        value = self._job_descriptor.job_metadata.get(field)
    elif field == 'script':
      value = self._try_op_to_script_body()
    elif field == 'create-time' or field == 'start-time':
      # TODO: Does Batch offer a start or end-time?
      # Check http://shortn/_FPYmD1weUF
      ds = google_batch_operations.get_create_time(self._op)
      value = google_base.parse_rfc3339_utc_string(ds)
    elif field == 'end-time' or field == 'last-update':
      # TODO: Does Batch offer an end-time?
      # Check http://shortn/_FPYmD1weUF
      ds = google_batch_operations.get_update_time(self._op)
      if ds:
        value = google_base.parse_rfc3339_utc_string(ds)
    elif field == 'status':
      value = self._operation_status()
    elif field == 'status-message':
      msg, _, _ = self._operation_status_message()
      value = msg
    elif field == 'status-detail':
      # As much detail as we can reasonably get from the operation
      msg, _, detail = self._operation_status_message()
      if detail:
        msg = detail
      value = msg
    else:
      raise ValueError(f'Unsupported field: "{field}"')

    return value if value else default

  def _try_op_to_script_body(self):
    env = self._get_task_environment()
    script = env.get(
        self._task_environment_name('prepare', google_utils.SCRIPT_VARNAME)
    )
    if not script:
      env = google_batch_operations.get_environment(self._op, _PREPARE_INDEX)
      script = env.get(google_utils.SCRIPT_VARNAME) if env else None
    if script:
      return ast.literal_eval(script)

  def _operation_status(self):
    """Returns the status of this operation.

    Raises:
      ValueError: if the operation status cannot be determined.

    Returns:
      A printable status string (RUNNING, SUCCESS, CANCELED or FAILURE).
    """
    if self._task:
      state = google_batch_operations.get_task_state(self._task)
      if state == batch_v1.TaskStatus.State.SUCCEEDED:
        return 'SUCCESS'
      if state == batch_v1.TaskStatus.State.FAILED:
        return 'FAILURE'
      if state == batch_v1.TaskStatus.State.UNEXECUTED:
        if google_batch_operations.is_canceled(self._op):
          return 'CANCELED'
        if (
            google_batch_operations.get_job_state(self._op)
            == batch_v1.JobStatus.State.CANCELLATION_IN_PROGRESS
        ):
          return 'RUNNING'
        return 'FAILURE'
      if state in [
          batch_v1.TaskStatus.State.STATE_UNSPECIFIED,
          batch_v1.TaskStatus.State.PENDING,
          batch_v1.TaskStatus.State.ASSIGNED,
          batch_v1.TaskStatus.State.RUNNING,
      ]:
        return 'RUNNING'
    else:
      if not google_batch_operations.is_done(self._op):
        return 'RUNNING'
      if google_batch_operations.is_success(self._op):
        return 'SUCCESS'
      if google_batch_operations.is_canceled(self._op):
        return 'CANCELED'
      if google_batch_operations.is_failed(self._op):
        return 'FAILURE'

    raise ValueError(
        'Status for operation {} could not be determined'.format(
            self._op['name']
        )
    )

  def _operation_status_message(self):
    """Returns the most relevant status string and failed action.

    This string is meant for display only.

    Returns:
      A triple of:
      - printable status message
      - the action that failed (if any)
      - a detail message (if available)
    """
    msg = ''
    action = None
    detail = None
    status_events = google_batch_operations.get_status_events(
        self._task or self._op
    )
    msg = self._operation_status()

    if status_events:
      detail = status_events[-1].description
    if self._task is None:
      detail = (
          'Batch task {} is not yet visible; using parent Job status: {}. {}'
      ).format(self._task_index, msg, detail or '').strip()
    return msg, action, detail


class GoogleBatchBatchHandler(object):
  """Implement the HttpBatch interface to enable simple serial batches."""

  def __init__(self, callback):
    self._cancel_list = []
    self._response_handler = callback

  def add(self, cancel_fn, request_id):
    self._cancel_list.append((request_id, cancel_fn))

  def execute(self):
    for request_id, cancel_fn in self._cancel_list:
      response = None
      exception = None
      try:
        response = cancel_fn.result()
      except:  # pylint: disable=bare-except
        exception = sys.exc_info()[1]

      self._response_handler(request_id, response, exception)


class GoogleBatchJobProvider(google_utils.GoogleJobProviderBase):
  """dsub provider implementation managing Jobs on Google Cloud."""

  def __init__(
      self, dry_run: bool, project: str, location: str, credentials=None
  ):
    storage_service = dsub_util.get_storage_service(credentials=credentials)

    self._dry_run = dry_run
    self._location = location
    self._project = project
    self._storage_service = storage_service

  def _batch_handler_def(self):
    return GoogleBatchBatchHandler

  def _operations_cancel_api_def(self):
    return batch_v1.BatchServiceClient().cancel_job

  def _get_provisioning_model(self, task_resources):
    if task_resources.preemptible:
      return batch_v1.AllocationPolicy.ProvisioningModel.SPOT
    else:
      return batch_v1.AllocationPolicy.ProvisioningModel.STANDARD

  def _get_batch_job_regions(self, regions, zones) -> List[str]:
    """Returns the list of regions and zones to use for a Batch Job request.

    If neither regions nor zones were specified for the Job, then use the
    Batch Job API location as the default region.

    Regions need to be prefixed with "regions/" and zones need to be prefixed
    with "zones/" as documented in
    https://cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs#LocationPolicy

    Args:
      regions (str): A space separated list of regions to use for the Job.
      zones (str): A space separated list of zones to use for the Job.
    """
    if regions:
      regions = [f'regions/{region}' for region in regions]
    if zones:
      zones = [f'zones/{zone}' for zone in zones]
    if not regions and not zones:
      return [f'regions/{self._location}']
    return (regions or []) + (zones or [])

  def _get_logging_env(self, logging_uri, user_project, include_filter_script):
    """Returns the environment for actions that copy logging files."""
    if not logging_uri.endswith('.log'):
      raise ValueError('Logging URI must end in ".log": {}'.format(logging_uri))

    logging_prefix = logging_uri[: -len('.log')]
    env = {
        'LOGGING_PATH': '{}.log'.format(logging_prefix),
        'STDOUT_PATH': '{}-stdout.log'.format(logging_prefix),
        'STDERR_PATH': '{}-stderr.log'.format(logging_prefix),
        'USER_PROJECT': user_project,
    }
    if include_filter_script:
      env[_LOG_FILTER_VAR] = repr(_LOG_FILTER_PYTHON)

    return env

  def _format_batch_job_id(self, job_descriptor) -> str:
    """Return one Google Batch Job ID for a submitted dsub JobDescriptor."""
    batch_job_id = job_descriptor.job_metadata['job-id']

    # A dsub-managed retry is a new JobDescriptor because Google Batch Jobs
    # are immutable. Keep the original dsub job-id as a label, while making
    # the provider ID unique for that retried task.
    task_descriptor = job_descriptor.task_descriptors[0]
    task_attempt = task_descriptor.task_metadata.get('task-attempt') or 1
    if task_attempt > 1:
      task_id = task_descriptor.task_metadata.get('task-id') or 0
      suffix = f'-{task_id}-{task_attempt}'
      batch_job_id = batch_job_id[: 63 - len(suffix)].rstrip('-') + suffix

    return batch_job_id[:63].rstrip('-')

  @staticmethod
  def _task_environment_prefix(action):
    return _TASK_ENV_PREFIX.format(action=action.upper())

  @classmethod
  def _prefix_task_environment(cls, action, environment):
    prefix = cls._task_environment_prefix(action)
    return {prefix + name: value for name, value in environment.items()}

  @classmethod
  def _activate_task_environment(cls, action, staged_manifest=False):
    """Build Bash that exposes one runnable's namespaced task variables."""
    prefix = cls._task_environment_prefix(action)
    command = ''
    if staged_manifest:
      command = f'source "{_TASK_ENV_FILE}"\n'
    return command + (
        f'for _dsub_env_name in "${{!{prefix}@}}"; do\n'
        f'  export "${{_dsub_env_name#{prefix}}}=${{!_dsub_env_name}}"\n'
        'done\n'
    )

  def _build_task_environment(self, task_view):
    """Build the TaskGroup environment for one dsub task."""
    job_params = task_view.job_params
    task_descriptor = task_view.task_descriptors[0]
    task_params = task_descriptor.task_params
    task_resources = task_descriptor.task_resources
    user_project = task_view.job_metadata['user-project'] or ''
    script = task_view.job_metadata['script']

    envs = job_params['envs'] | task_params['envs']
    inputs = job_params['inputs'] | task_params['inputs']
    outputs = job_params['outputs'] | task_params['outputs']
    mounts = job_params['mounts']

    # User variables stay unprefixed and are consumed by the user runnable.
    # Internal variables are namespaced by runnable and activated by a short
    # Bash preamble. This prevents user variables from shadowing dsub's input,
    # output, metadata, or logging controls.
    environment = self._build_user_environment(
        envs, inputs, outputs, mounts, _DATA_MOUNT_POINT
    )
    action_environments = {
        'continuous_logging': self._get_logging_env(
            task_resources.logging_path.uri, user_project, True
        ),
        'prepare': self._get_prepare_env(
            script, task_view, inputs, outputs, mounts, _DATA_MOUNT_POINT
        ),
        'localization': self._get_localization_env(
            inputs, user_project, _DATA_MOUNT_POINT
        ),
        'delocalization': self._get_delocalization_env(
            outputs, user_project, _DATA_MOUNT_POINT
        ),
        'final_logging': self._get_logging_env(
            task_resources.logging_path.uri, user_project, False
        ),
    }
    for action, action_environment in action_environments.items():
      environment.update(
          self._prefix_task_environment(action, action_environment)
      )

    return google_batch_operations.build_environment(environment)

  @staticmethod
  def _gcs_object_parts(uri):
    if not uri.startswith('gs://'):
      raise ValueError(f'Expected a GCS URI, got: {uri}')
    bucket, separator, object_name = uri[len('gs://') :].partition('/')
    if not bucket or not separator or not object_name:
      raise ValueError(f'Expected a GCS object URI, got: {uri}')
    return bucket, object_name

  def _task_manifest_uris(self, job_descriptor):
    logging_uri = (
        job_descriptor.task_descriptors[0].task_resources.logging_path.uri
    )
    bucket, _ = self._gcs_object_parts(logging_uri)
    batch_job_id = self._format_batch_job_id(job_descriptor)
    prefix = f'gs://{bucket}/.dsub/google-batch/{batch_job_id}/tasks'
    return f'{prefix}.jsonl', f'{prefix}.index'

  def _build_task_manifest(self, job_descriptor):
    """Serialize task environments with a fixed-width random-access index."""
    uri, index_uri = self._task_manifest_uris(job_descriptor)
    common_environment = None
    for task_view in job_model.task_view_generator(job_descriptor):
      environment = dict(self._build_task_environment(task_view).variables)
      if common_environment is None:
        common_environment = environment
      else:
        common_environment = {
            name: value
            for name, value in common_environment.items()
            if environment.get(name) == value
        }

    common_environment = common_environment or {}
    data = bytearray()
    index = bytearray()
    for task_view in job_model.task_view_generator(job_descriptor):
      environment = dict(self._build_task_environment(task_view).variables)
      environment = {
          name: value
          for name, value in environment.items()
          if name not in common_environment
      }
      record = (
          json.dumps(environment, sort_keys=True, separators=(',', ':')) + '\n'
      ).encode('utf-8')
      start = len(data)
      end = start + len(record) - 1
      index_record = f'{start:020d} {end:020d}\n'.encode('ascii')
      if len(index_record) != _TASK_MANIFEST_INDEX_RECORD_SIZE:
        raise ValueError('Task manifest exceeded its supported offset size')
      data.extend(record)
      index.extend(index_record)

    return _TaskManifest(
        uri=uri,
        index_uri=index_uri,
        user_project=job_descriptor.job_metadata['user-project'] or '',
        data=bytes(data),
        index=bytes(index),
        common_environment=common_environment,
    )

  def _upload_gcs_object(self, uri, contents, content_type, user_project):
    bucket, object_name = self._gcs_object_parts(uri)
    media = googleapiclient.http.MediaIoBaseUpload(
        io.BytesIO(contents), mimetype=content_type, resumable=True
    )
    request_args = {
        'bucket': bucket,
        'name': object_name,
        'media_body': media,
    }
    if user_project:
      request_args['userProject'] = user_project
    # googleapiclient's dynamically generated Resource exposes objects().
    # pylint: disable=no-member
    self._storage_service.objects().insert(**request_args).execute(
        num_retries=5
    )
    # pylint: enable=no-member

  def _stage_task_manifest(self, manifest):
    self._upload_gcs_object(
        manifest.uri,
        manifest.data,
        'application/x-ndjson',
        manifest.user_project,
    )
    self._upload_gcs_object(
        manifest.index_uri,
        manifest.index,
        'text/plain',
        manifest.user_project,
    )

  def _download_gcs_object(self, uri, user_project):
    bucket, object_name = self._gcs_object_parts(uri)
    request_args = {'bucket': bucket, 'object': object_name}
    if user_project:
      request_args['userProject'] = user_project
    # pylint: disable=no-member
    request = self._storage_service.objects().get_media(**request_args)
    # pylint: enable=no-member
    destination = io.BytesIO()
    downloader = googleapiclient.http.MediaIoBaseDownload(
        destination, request, chunksize=1024 * 1024
    )
    done = False
    while not done:
      _, done = downloader.next_chunk(num_retries=5)
    return destination.getvalue()

  def _download_task_manifest(self, job):
    environment = google_batch_operations.get_task_spec_environment(job)
    uri = environment.get(_TASK_MANIFEST_URI_VARNAME)
    if not uri:
      return None
    user_project = environment.get(_TASK_MANIFEST_USER_PROJECT_VARNAME, '')
    contents = self._download_gcs_object(uri, user_project).decode('utf-8')
    common_environment = dict(environment)
    for name in [
        _TASK_MANIFEST_URI_VARNAME,
        _TASK_MANIFEST_INDEX_URI_VARNAME,
        _TASK_MANIFEST_INDEX_RECORD_SIZE_VARNAME,
        _TASK_MANIFEST_USER_PROJECT_VARNAME,
    ]:
      common_environment.pop(name, None)
    task_environments = []
    for line in contents.splitlines():
      task_environment = dict(common_environment)
      task_environment.update(json.loads(line))
      task_environments.append(task_environment)
    return task_environments

  def _get_gcs_volumes(self, mounts) -> List[batch_v1.types.Volume]:
    # Return a list of GCS volumes for the Batch Job request.
    gcs_volumes = []
    for gcs_mount in param_util.get_gcs_mounts(mounts):
      mount_path = os.path.join(_VOLUME_MOUNT_POINT, gcs_mount.docker_path)
      # Normalize mount path because API does not allow trailing slashes
      normalized_mount_path = os.path.normpath(mount_path)
      gcs_volume = google_batch_operations.build_gcs_volume(
          gcs_mount.value[len('gs://') :], normalized_mount_path, ['-o ro']
      )
      gcs_volumes.append(gcs_volume)
    return gcs_volumes

  def _get_gcs_volumes_for_user_command(self, mounts) -> List[str]:
    # Return a list of GCS volumes to be included with the
    # user-command runnable
    user_command_volumes = []
    for gcs_mount in param_util.get_gcs_mounts(mounts):
      volume_mount_point = os.path.normpath(
          os.path.join(_VOLUME_MOUNT_POINT, gcs_mount.docker_path)
      )
      data_mount_point = os.path.normpath(
          os.path.join(_DATA_MOUNT_POINT, gcs_mount.docker_path)
      )
      user_command_volumes.append(f'{volume_mount_point}:{data_mount_point}')
    return user_command_volumes

  def _create_batch_request(
      self,
      job_descriptor: job_model.JobDescriptor,
      task_manifest=None,
  ):
    job_metadata = job_descriptor.job_metadata
    job_params = job_descriptor.job_params
    job_resources = job_descriptor.job_resources
    task_resources = job_descriptor.task_descriptors[0].task_resources

    # Set up VM-specific variables
    datadisk_volume = google_batch_operations.build_volume(
        disk=google_utils.DATA_DISK_NAME, path=_VOLUME_MOUNT_POINT
    )

    # Google Batch labels apply to the whole Job. Task-specific labels remain
    # in each serialized task descriptor and are filtered client-side.
    # pylint: disable=g-complex-comprehension
    labels = {
        label.name: label.value if label.value else ''
        for label in google_base.build_pipeline_labels(
            job_metadata, {}
        )
        | job_params['labels']
    }
    # pylint: enable=g-complex-comprehension

    # Set local variables for the core pipeline values
    script = job_descriptor.job_metadata['script']

    # Track 0-based runnable indexes for cross-task awareness
    user_action = 4 if task_manifest else 3

    continuous_logging_cmd = _CONTINUOUS_LOGGING_CMD.format(
        log_msg_fn=google_utils.LOG_MSG_FN,
        gsutil_cp_fn=google_utils.GSUTIL_CP_FN,
        log_filter_var=_LOG_FILTER_VAR,
        log_filter_script_path=_LOG_FILTER_SCRIPT_PATH,
        python_decode_script=google_utils.PYTHON_DECODE_SCRIPT,
        logging_dir=_LOGGING_DIR,
        log_cp=_LOG_CP.format(
            log_filter_script_path=_LOG_FILTER_SCRIPT_PATH,
            user_action=user_action,
        ),
        log_interval=job_resources.log_interval or '60s',
    )

    logging_cmd = _FINAL_LOGGING_CMD.format(
        log_msg_fn=google_utils.LOG_MSG_FN,
        gsutil_cp_fn=google_utils.GSUTIL_CP_FN,
        log_filter_var=_LOG_FILTER_VAR,
        log_filter_script_path=_LOG_FILTER_SCRIPT_PATH,
        python_decode_script=google_utils.PYTHON_DECODE_SCRIPT,
        logging_dir=_LOGGING_DIR,
        log_cp=_LOG_CP.format(
            log_filter_script_path=_LOG_FILTER_SCRIPT_PATH,
            user_action=user_action,
        ),
    )

    # Set up command and environments for the prepare, localization, user,
    # and de-localization actions
    script_path = os.path.join(_SCRIPT_DIR, script.name)
    prepare_command = google_utils.PREPARE_CMD.format(
        log_msg_fn=google_utils.LOG_MSG_FN,
        mk_runtime_dirs=google_utils.make_runtime_dirs_command(
            _SCRIPT_DIR, _TMP_DIR, _WORKING_DIR
        ),
        script_var=google_utils.SCRIPT_VARNAME,
        python_decode_script=google_utils.PYTHON_DECODE_SCRIPT,
        script_path=script_path,
        mk_io_dirs=google_utils.MK_IO_DIRS,
    )
    # pylint: disable=line-too-long

    mounts = job_params['mounts']
    gcs_volumes = self._get_gcs_volumes(mounts)
    runnable_environment = google_batch_operations.build_environment({})
    task_spec_environment = None
    if task_manifest:
      shared_environment = dict(task_manifest.common_environment)
      shared_environment.update(
          {
              _TASK_MANIFEST_URI_VARNAME: task_manifest.uri,
              _TASK_MANIFEST_INDEX_URI_VARNAME: task_manifest.index_uri,
              _TASK_MANIFEST_INDEX_RECORD_SIZE_VARNAME: str(
                  _TASK_MANIFEST_INDEX_RECORD_SIZE
              ),
              _TASK_MANIFEST_USER_PROJECT_VARNAME: (
                  task_manifest.user_project
              ),
          }
      )
      task_spec_environment = google_batch_operations.build_environment(
          shared_environment
      )
    else:
      if len(job_descriptor.task_descriptors) != 1:
        raise ValueError(
            'Multi-task google-batch jobs require an indexed task manifest'
        )
      task_spec_environment = self._build_task_environment(job_descriptor)

    # Build the list of runnables (aka actions)
    runnables = []

    if task_manifest:
      runnables.append(
          # Select this task's environment using BATCH_TASK_INDEX.
          google_batch_operations.build_runnable(
              run_in_background=False,
              always_run=False,
              image_uri=google_utils.CLOUD_SDK_IMAGE,
              environment=runnable_environment,
              entrypoint='/bin/bash',
              volumes=[f'{_VOLUME_MOUNT_POINT}:{_DATA_MOUNT_POINT}'],
              commands=['-c', _TASK_MANIFEST_BOOTSTRAP_CMD],
          )
      )

    runnables.append(
        # logging
        google_batch_operations.build_runnable(
            run_in_background=True,
            always_run=False,
            image_uri=google_utils.CLOUD_SDK_IMAGE,
            environment=runnable_environment,
            entrypoint='/bin/bash',
            volumes=[f'{_VOLUME_MOUNT_POINT}:{_DATA_MOUNT_POINT}'],
            commands=[
                '-c',
                self._activate_task_environment(
                    'continuous_logging', bool(task_manifest)
                )
                + continuous_logging_cmd,
            ],
        )
    )

    runnables.append(
        # prepare
        google_batch_operations.build_runnable(
            run_in_background=False,
            always_run=False,
            image_uri=google_utils.CLOUD_SDK_IMAGE,
            environment=runnable_environment,
            entrypoint='/bin/bash',
            volumes=[f'{_VOLUME_MOUNT_POINT}:{_DATA_MOUNT_POINT}'],
            commands=[
                '-c',
                self._activate_task_environment(
                    'prepare', bool(task_manifest)
                )
                + prepare_command,
            ],
        )
    )

    runnables.append(
        # localization
        google_batch_operations.build_runnable(
            run_in_background=False,
            always_run=False,
            image_uri=google_utils.CLOUD_SDK_IMAGE,
            environment=runnable_environment,
            entrypoint='/bin/bash',
            volumes=[f'{_VOLUME_MOUNT_POINT}:{_DATA_MOUNT_POINT}'],
            commands=[
                '-c',
                self._activate_task_environment(
                    'localization', bool(task_manifest)
                )
                + google_utils.LOCALIZATION_CMD.format(
                    log_msg_fn=google_utils.LOG_MSG_FN,
                    recursive_cp_fn=google_utils.GSUTIL_RSYNC_FN,
                    cp_fn=google_utils.GSUTIL_CP_FN,
                    cp_loop=google_utils.LOCALIZATION_LOOP,
                ),
            ],
        )
    )

    user_command_volumes = [f'{_VOLUME_MOUNT_POINT}:{_DATA_MOUNT_POINT}']
    for gcs_volume in self._get_gcs_volumes_for_user_command(mounts):
      user_command_volumes.append(gcs_volume)
    runnables.append(
        # user-command
        google_batch_operations.build_runnable(
            run_in_background=False,
            always_run=False,
            image_uri=job_resources.image,
            environment=runnable_environment,
            entrypoint='/usr/bin/env',
            volumes=user_command_volumes,
            commands=[
                'bash',
                '-c',
                (
                    f'source "{_TASK_ENV_FILE}"\n' if task_manifest else ''
                )
                + google_utils.USER_CMD.format(
                    tmp_dir=_TMP_DIR,
                    working_dir=_WORKING_DIR,
                    user_script=script_path,
                ),
            ],
        )
    )

    runnables.append(
        # delocalization
        google_batch_operations.build_runnable(
            run_in_background=False,
            always_run=False,
            image_uri=google_utils.CLOUD_SDK_IMAGE,
            environment=runnable_environment,
            entrypoint='/bin/bash',
            volumes=[f'{_VOLUME_MOUNT_POINT}:{_DATA_MOUNT_POINT}:ro'],
            commands=[
                '-c',
                self._activate_task_environment(
                    'delocalization', bool(task_manifest)
                )
                + google_utils.LOCALIZATION_CMD.format(
                    log_msg_fn=google_utils.LOG_MSG_FN,
                    recursive_cp_fn=google_utils.GSUTIL_RSYNC_FN,
                    cp_fn=google_utils.GSUTIL_CP_FN,
                    cp_loop=google_utils.DELOCALIZATION_LOOP,
                ),
            ],
        )
    )

    runnables.append(
        # final_logging
        google_batch_operations.build_runnable(
            run_in_background=False,
            always_run=True,
            image_uri=google_utils.CLOUD_SDK_IMAGE,
            environment=runnable_environment,
            entrypoint='/bin/bash',
            volumes=[f'{_VOLUME_MOUNT_POINT}:{_DATA_MOUNT_POINT}'],
            commands=[
                '-c',
                self._activate_task_environment(
                    'final_logging', bool(task_manifest)
                )
                + logging_cmd,
            ],
        ),
    )

    # Prepare the VM (resources) configuration. The InstancePolicy describes an
    # instance type and resources attached to each VM. The AllocationPolicy
    # describes when, where, and how compute resources should be allocated
    # for the Job.
    boot_disk_size = (
        job_resources.boot_disk_size if job_resources.boot_disk_size else 0
    )
    boot_disk = google_batch_operations.build_persistent_disk(
        size_gb=max(boot_disk_size, job_model.LARGE_BOOT_DISK_SIZE),
        disk_type=job_model.DEFAULT_DISK_TYPE,
    )
    disk = google_batch_operations.build_persistent_disk(
        size_gb=job_resources.disk_size,
        disk_type=job_resources.disk_type or job_model.DEFAULT_DISK_TYPE,
    )
    attached_disk = google_batch_operations.build_attached_disk(
        disk=disk, device_name=google_utils.DATA_DISK_NAME
    )

    if job_resources.machine_type:
      machine_type = job_resources.machine_type
    elif job_resources.min_cores or job_resources.min_ram:
      machine_type = (
          google_custom_machine.GoogleCustomMachine.build_machine_type(
              job_resources.min_cores, job_resources.min_ram
          )
      )
    else:
      machine_type = job_model.DEFAULT_MACHINE_TYPE

    instance_policy = google_batch_operations.build_instance_policy(
        boot_disk=boot_disk,
        disks=attached_disk,
        machine_type=machine_type,
        accelerators=google_batch_operations.build_accelerators(
            accelerator_type=job_resources.accelerator_type,
            accelerator_count=job_resources.accelerator_count,
        ),
        provisioning_model=self._get_provisioning_model(task_resources),
    )

    ipt = google_batch_operations.build_instance_policy_or_template(
        instance_policy=instance_policy,
        install_gpu_drivers=True
        if job_resources.accelerator_type is not None
        else False,
    )

    if job_resources.service_account:
      scopes = job_resources.scopes or google_base.DEFAULT_SCOPES
      service_account = google_batch_operations.build_service_account(
          service_account_email=job_resources.service_account, scopes=scopes
      )
    else:
      service_account = None

    network_policy = google_batch_operations.build_network_policy(
        network=job_resources.network,
        subnetwork=job_resources.subnetwork,
        no_external_ip_address=job_resources.use_private_address,
    )

    location_policy = google_batch_operations.build_location_policy(
        allowed_locations=self._get_batch_job_regions(
            regions=job_resources.regions, zones=job_resources.zones
        ),
    )

    allocation_policy = google_batch_operations.build_allocation_policy(
        ipts=[ipt],
        service_account=service_account,
        network_policy=network_policy,
        location_policy=location_policy,
    )

    logs_policy = google_batch_operations.build_logs_policy(
        # Explicitly end the logging path with a slash.
        # This will prompt Batch API to create the log, stdout, and stderr
        # files in the specified directory.
        batch_v1.LogsPolicy.Destination.PATH,
        _BATCH_LOG_DIR + '/',
    )

    # Bring together the task definition(s) and build the Job request.
    task_spec = google_batch_operations.build_task_spec(
        runnables=runnables,
        volumes=([datadisk_volume] + gcs_volumes),
        max_run_duration=job_resources.timeout,
        environment=task_spec_environment,
    )
    task_group = google_batch_operations.build_task_group(
        task_spec,
        task_count_per_node=1,
        task_count=len(job_descriptor.task_descriptors),
    )

    job = google_batch_operations.build_job(
        [task_group], allocation_policy, labels, logs_policy
    )

    batch_job_id = self._format_batch_job_id(job_descriptor)

    job_request = batch_v1.CreateJobRequest(
        parent=f'projects/{self._project}/locations/{self._location}',
        job=job,
        job_id=batch_job_id,
    )
    # pylint: enable=line-too-long
    return job_request

  def _submit_batch_job(self, request):
    client = batch_v1.BatchServiceClient()
    job_response = client.create_job(request=request)
    print(f'Provider internal-id (operation): {job_response.name}')

  def _prepare_batch_request(self, job_descriptor):
    """Build one Batch request and its optional indexed task manifest."""
    if len(job_descriptor.task_descriptors) > 1:
      task_manifest = self._build_task_manifest(job_descriptor)
      return (
          self._create_batch_request(job_descriptor, task_manifest),
          task_manifest,
      )

    request = self._create_batch_request(job_descriptor)
    if batch_v1.Job.pb(request.job).ByteSize() <= (
        _MAX_INLINE_JOB_DEFINITION_BYTES
    ):
      return request, None

    # An unusually large single-task definition still needs to stay below
    # Batch's job definition limit, so use the indexed representation.
    task_manifest = self._build_task_manifest(job_descriptor)
    return (
        self._create_batch_request(job_descriptor, task_manifest),
        task_manifest,
    )

  def submit_job(
      self,
      job_descriptor: job_model.JobDescriptor,
      skip_if_output_present: bool,
  ) -> Dict[str, any]:
    # Validate task data and resources.
    param_util.validate_submit_args_or_fail(
        job_descriptor,
        provider_name=_PROVIDER_NAME,
        input_providers=_SUPPORTED_INPUT_PROVIDERS,
        output_providers=_SUPPORTED_OUTPUT_PROVIDERS,
        logging_providers=_SUPPORTED_LOGGING_PROVIDERS,
    )

    # Select tasks before creating the single provider Job. Skipped tasks are
    # omitted from the TaskGroup rather than producing separate provider Jobs.
    task_descriptors = []
    for task_descriptor in job_descriptor.task_descriptors:
      job_params = job_descriptor.job_params
      task_params = task_descriptor.task_params

      outputs = job_params['outputs'] | task_params['outputs']
      if skip_if_output_present:
        # check whether the output's already there
        if dsub_util.outputs_are_present(outputs, self._storage_service):
          print('Skipping task because its outputs are present')
          continue

      task_descriptors.append(task_descriptor)

    if not task_descriptors:
      return {'job-id': dsub_util.NO_JOB}

    if len(task_descriptors) > _MAX_TASKS_PER_JOB:
      raise ValueError(
          'google-batch supports at most {} tasks in one job; got {}'.format(
              _MAX_TASKS_PER_JOB, len(task_descriptors)
          )
      )

    provisioning_models = {
        bool(task_descriptor.task_resources.preemptible)
        for task_descriptor in task_descriptors
    }
    if len(provisioning_models) > 1:
      raise ValueError(
          'All tasks in a google-batch job must use the same provisioning '
          'model'
      )

    launch_descriptor = job_model.JobDescriptor(
        job_descriptor.job_metadata,
        job_descriptor.job_params,
        job_descriptor.job_resources,
        task_descriptors,
    )
    request, task_manifest = self._prepare_batch_request(launch_descriptor)
    if self._dry_run:
      # CreateJobRequest's repr resembles YAML but is not YAML serializable.
      print([request])
    else:
      if task_manifest:
        self._stage_task_manifest(task_manifest)
      self._submit_batch_job(request)

    return {
        'job-id': job_descriptor.job_metadata['job-id'],
        'user-id': job_descriptor.job_metadata.get('user-id'),
        'task-id': [
            task_descriptor.task_metadata.get('task-id')
            for task_descriptor in task_descriptors
            if task_descriptor.task_metadata.get('task-id') is not None
        ],
    }

  def delete_jobs(
      self,
      user_ids,
      job_ids,
      task_ids,
      labels,
      create_time_min=None,
      create_time_max=None,
  ):
    """Kills the operations associated with the specified job or job.task.

    Args:
      user_ids: List of user ids who "own" the job(s) to cancel.
      job_ids: List of job_ids to cancel.
      task_ids: List of task-ids to cancel.
      labels: List of LabelParam, each must match the job(s) to be canceled.
      create_time_min: a timezone-aware datetime value for the earliest create
        time of a task, inclusive.
      create_time_max: a timezone-aware datetime value for the most recent
        create time of a task, inclusive.

    Returns:
      A list of tasks canceled and a list of error messages.
    """
    if task_ids and set(task_ids) != {'*'}:
      raise ValueError(
          'google-batch cannot cancel an individual task in a multi-task '
          'job; cancel the dsub job instead'
      )

    # Look up the job(s)
    tasks = list(
        self.lookup_job_tasks(
            {'RUNNING'},
            user_ids=user_ids,
            job_ids=job_ids,
            task_ids=task_ids,
            labels=labels,
            create_time_min=create_time_min,
            create_time_max=create_time_max,
        )
    )

    print('Found %d tasks to delete.' % len(tasks))
    tasks_by_job = {}
    for task in tasks:
      tasks_by_job.setdefault(task.get_field('internal-id'), []).append(task)

    canceled_jobs, error_messages = google_base.cancel(
        self._batch_handler_def(),
        self._operations_cancel_api_def(),
        [job_tasks[0] for job_tasks in tasks_by_job.values()],
    )
    canceled_job_ids = {
        task.get_field('internal-id') for task in canceled_jobs
    }
    canceled_tasks = [
        task
        for job_id, job_tasks in tasks_by_job.items()
        if job_id in canceled_job_ids
        for task in job_tasks
    ]
    return canceled_tasks, error_messages

  @staticmethod
  def _matches_task_filters(
      task,
      statuses,
      user_ids,
      job_ids,
      job_names,
      task_ids,
      task_attempts,
      labels,
      create_time_min,
      create_time_max,
  ):
    def normalize(values):
      if not values or set(values) == {'*'}:
        return None
      return {str(value) for value in values}

    statuses = normalize(statuses)
    user_ids = normalize(user_ids)
    job_ids = normalize(job_ids)
    job_names = normalize(job_names)
    task_ids = normalize(task_ids)
    task_attempts = normalize(task_attempts)

    if statuses and task.get_field('task-status') not in statuses:
      return False
    if user_ids and str(task.get_field('user-id')) not in user_ids:
      return False
    if job_ids and str(task.get_field('job-id')) not in job_ids:
      return False
    if job_names and str(task.get_field('job-name')) not in job_names:
      return False
    if task_ids and str(task.get_field('task-id')) not in task_ids:
      return False
    if (
        task_attempts
        and str(task.get_field('task-attempt')) not in task_attempts
    ):
      return False

    task_labels = task.get_field('labels', {})
    if labels and not all(
        label.name in task_labels
        and task_labels[label.name] == label.value
        for label in labels
    ):
      return False

    create_time = task.get_field('create-time')
    if create_time_min and create_time < create_time_min:
      return False
    if create_time_max and create_time > create_time_max:
      return False
    return True

  def lookup_job_tasks(
      self,
      statuses: Set[str],
      user_ids=None,
      job_ids=None,
      job_names=None,
      task_ids=None,
      task_attempts=None,
      labels=None,
      create_time_min=None,
      create_time_max=None,
      max_tasks=0,
      page_size=0,
  ):
    client = batch_v1.BatchServiceClient()
    # Batch can bind its default size (500) into a continuation token while
    # the Python pager retains the request's zero value. Send an explicit
    # size on the first request so every page uses the same value.
    effective_page_size = page_size or _DEFAULT_LIST_PAGE_SIZE
    # Job-level filters can be sent to list_jobs. Status and task metadata
    # filters must be evaluated after list_tasks: a failed multi-task Job can
    # still contain successful tasks, and task labels are not Job labels.
    ops_filter = self._build_query_filter(
        {'*'},
        user_ids,
        job_ids,
        job_names,
        None,
        None,
        None,
        create_time_min,
        create_time_max,
    )
    # Initialize request argument(s)
    request = batch_v1.ListJobsRequest(
        parent=f'projects/{self._project}/locations/{self._location}',
        filter=ops_filter,
        page_size=effective_page_size,
    )

    jobs = list(client.list_jobs(request=request))
    # An exact Job lookup can succeed before the list index exposes a newly
    # submitted Job. Retain label-based listing for legacy IDs and retries.
    found_job_ids = {job.labels.get('job-id') for job in jobs}
    for job_id in set(job_ids or []) - found_job_ids:
      if job_id == '*':
        continue
      try:
        job = client.get_job(request=batch_v1.GetJobRequest(
            name=f'{request.parent}/jobs/{job_id}'
        ))
      except google_exceptions.NotFound:
        continue
      jobs.append(job)

    operations = []
    for job in jobs:
      manifest_environments = self._download_task_manifest(job)
      task_request = batch_v1.ListTasksRequest(
          parent=f'{job.name}/taskGroups/{_TASK_GROUP_NAME}',
          page_size=effective_page_size,
      )
      try:
        batch_tasks = {
            GoogleBatchOperation._get_task_index(task): task
            for task in client.list_tasks(request=task_request)
        }
      except google_exceptions.NotFound:
        # The parent Job is visible, but its TaskGroup may still be pending.
        batch_tasks = {}
      task_count = job.task_groups[0].task_count or 1
      # A missing task must not disappear from dsub's wait loop. Until Batch
      # exposes its Task record, use the Job's status and the submitted task
      # metadata. Always prefer actual Task records, including mixed outcomes.
      for task_index in sorted(set(range(task_count)) | set(batch_tasks)):
        batch_task = batch_tasks.get(task_index)
        task_environment = None
        if manifest_environments is not None:
          if task_index >= len(manifest_environments):
            raise ValueError(
                'Google Batch task index {} is missing from manifest {}'.format(
                    task_index,
                    google_batch_operations.get_task_spec_environment(job).get(
                        _TASK_MANIFEST_URI_VARNAME
                    ),
                )
            )
          task_environment = manifest_environments[task_index]
        task = GoogleBatchOperation(
            job, batch_task, task_environment, task_index=task_index
        )
        if self._matches_task_filters(
            task,
            statuses,
            user_ids,
            job_ids,
            job_names,
            task_ids,
            task_attempts,
            labels,
            create_time_min,
            create_time_max,
        ):
          operations.append(task)

    # Sort tasks by Job create-time, then by dsub task-id, matching the other
    # providers' newest-job-first behavior.
    def task_sort_key(operation):
      return (
          operation.get_field('create-time'),
          int(operation.get_field('task-id', 0)),
          int(operation.get_field('task-attempt', 0)),
      )

    operations.sort(key=task_sort_key, reverse=True)
    for task_number, operation in enumerate(operations, start=1):
      yield operation
      if 0 < max_tasks <= task_number:
        return

  def get_tasks_completion_messages(self, tasks):
    return [task.get_field('status-detail') for task in tasks]
