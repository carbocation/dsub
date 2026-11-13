# Copyright 2026 Google Inc. All Rights Reserved.
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
"""Execute helper commands with only the Cloud SDK's bundled Python."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from dsub.lib import param_util
from dsub.providers import google_batch
from dsub.providers import google_utils
from dsub.providers import google_v2_base
from test.unit import google_batch_test


class GooglePythonFallbackTest(unittest.TestCase):

  _GCLOUD_SCRIPT = r"""
import json
import os
from pathlib import Path
import sys

assert os.environ['CLOUDSDK_PYTHON'] == os.environ['EXPECTED_BUNDLED_PYTHON']
args = sys.argv[1:]
with open(os.environ['GCLOUD_CALLS'], 'a', encoding='utf-8') as output:
  output.write(json.dumps(args) + '\n')
assert args[:1] == ['storage'], args
if args[1] == 'cat':
  byte_range = next(arg for arg in args if arg.startswith('--range='))
  start, end = map(int, byte_range.split('=', 1)[1].split('-'))
  data = (Path(os.environ['GCS_ROOT']) / args[-1][len('gs://'):]).read_bytes()
  sys.stdout.buffer.write(data[start:end + 1])
else:
  assert args[1] in ['cp', 'rsync'], args
"""

  def setUp(self):
    directory = tempfile.TemporaryDirectory(prefix='dsub bundled python ')
    self.addCleanup(directory.cleanup)
    self.root = Path(directory.name)
    self.tools_dir = self.root / 'bin'
    self.tools_dir.mkdir()
    for command in ['mkdir', 'chmod', 'date', 'touch']:
      (self.tools_dir / command).symlink_to(shutil.which(command))
    self.bundled_python = self.root / 'bundled python' / 'python3'
    self.bundled_python.parent.mkdir()
    self.bundled_python.symlink_to(sys.executable)
    gcloud_body = self.root / 'gcloud.py'
    gcloud_body.write_text(self._GCLOUD_SCRIPT)
    gcloud = self.tools_dir / 'gcloud'
    gcloud.write_text(
        '#!/bin/bash\nexec "${CLOUDSDK_PYTHON:?}" ' +
        shlex.quote(str(gcloud_body)) + ' "$@"\n')
    gcloud.chmod(0o755)
    self.calls_path = self.root / 'gcloud-calls.jsonl'
    self.gcs_root = self.root / 'gcs'
    self.environment = dict(os.environ)
    self.environment.pop('CLOUDSDK_PYTHON', None)
    self.environment.update({
        'PATH': str(self.tools_dir),
        'GCLOUD_CALLS': str(self.calls_path),
        'GCS_ROOT': str(self.gcs_root),
        'EXPECTED_BUNDLED_PYTHON': str(self.bundled_python),
    })
    self.assertIsNone(shutil.which('python3', path=self.environment['PATH']))
    storage_patch = mock.patch.object(
        google_batch.dsub_util, 'get_storage_service')
    self.addCleanup(storage_patch.stop)
    storage_patch.start()
    self.provider = google_batch.GoogleBatchJobProvider(
        dry_run=False, project='test-project', location='us-central1')

  def _run(self, command, variables=None, data_dir=None):
    command = command.replace(google_utils.CLOUD_SDK_BUNDLED_PYTHON,
                              str(self.bundled_python))
    if data_dir is not None:
      command = command.replace('/mnt/data', str(data_dir))
    environment = dict(self.environment)
    environment.update(variables or {})
    if data_dir is not None:
      environment = {
          name: value.replace('/mnt/data', str(data_dir))
          for name, value in environment.items()
      }
    result = subprocess.run(
        ['/bin/bash', '-e', '-u', '-o', 'pipefail', '-c', command],
        env=environment, capture_output=True, text=True, timeout=10)
    self.assertEqual(result.returncode, 0, result.stderr)
    return result

  def test_fallback_runs_python_and_gcloud_without_system_python(self):
    result = self._run(
        google_utils.PYTHON_FN + '\npython3 -c "print(42)"\n' +
        'gcloud storage cp gs://bucket/input gs://bucket/output')
    self.assertEqual(result.stdout, '42\n')
    self.assertEqual(len(self.calls_path.read_text().splitlines()), 1)

  def test_existing_python_and_cloudsdk_override_are_preserved(self):
    (self.tools_dir / 'python3').symlink_to(sys.executable)
    result = self._run(
        google_utils.PYTHON_FN + '\ntype -t python3\n' +
        'python3 -c "import os; print(os.environ[\'CLOUDSDK_PYTHON\'])"',
        variables={'CLOUDSDK_PYTHON': 'operator-selected-python'})
    self.assertEqual(result.stdout, 'file\noperator-selected-python\n')

  def test_all_batch_helper_steps_run_with_bundled_python(self):
    for task_count in [1, 3]:
      with self.subTest(task_count=task_count):
        if self.calls_path.exists():
          self.calls_path.unlink()
        descriptor = google_batch_test.GoogleBatchProviderTest._job_descriptor(
            task_count=task_count)
        descriptor.job_params['inputs'].add(
            param_util.InputFileParamUtil('input').make_param(
                'INPUT', 'gs://test-bucket/input.txt', recursive=False))
        descriptor.job_params['outputs'].add(
            param_util.OutputFileParamUtil('output').make_param(
                'OUTPUT', 'gs://test-bucket/output.txt', recursive=False))
        request, manifest = self.provider._prepare_batch_request(descriptor)
        task_spec = request.job.task_groups[0].task_spec
        variables = dict(task_spec.environment.variables)
        variables['BATCH_TASK_INDEX'] = str(task_count - 1)
        data_dir = self.root / ('data-%s' % task_count)
        logging_dir = data_dir / '.logging'
        logging_dir.mkdir(parents=True)
        # Decode the continuous logging filter without waiting in its loop.
        (logging_dir / '.stop_logging').touch()
        if manifest:
          for uri, data in [(manifest.uri, manifest.data),
                            (manifest.index_uri, manifest.index)]:
            path = self.gcs_root / uri[len('gs://'):]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for runnable in task_spec.runnables:
          if runnable.container.entrypoint == '/bin/bash':
            self._run(runnable.container.commands[1], variables, data_dir)
        self.assertEqual(
            (data_dir / 'script' / 'script.sh').read_text(),
            descriptor.job_metadata['script'].value)
        self.assertEqual(
            (data_dir / '.log_filter_script.py').read_text(),
            google_batch._LOG_FILTER_PYTHON)
        calls = [json.loads(line)
                 for line in self.calls_path.read_text().splitlines()]
        self.assertEqual(sum(call[:2] == ['storage', 'cat'] for call in calls),
                         2 if manifest else 0)
        self.assertEqual(sum(call[:2] == ['storage', 'cp'] for call in calls), 5)
        if manifest:
          self.assertIn('export SAMPLE_ID=sample-3\n',
                        (data_dir / '.dsub' / 'task-environment.sh').read_text())

  def test_cls_logging_templates_initialize_gcloud_python(self):
    for template in [google_v2_base._FINAL_LOGGING_CMD,
                     google_v2_base._CONTINUOUS_LOGGING_CMD]:
      with self.subTest(template=template):
        # A pre-existing stop directory lets the continuous step return after
        # checking its interpreter, without waiting or accessing system paths.
        command = template.format(
            log_msg_fn=google_utils.LOG_MSG_FN,
            log_cp_fn=google_utils.LOG_CP_FN,
            python_fn=google_utils.PYTHON_FN,
            log_cp_cmd='gcloud storage cp gs://bucket/in gs://bucket/out',
            final_logging_action='done', log_interval='1')
        command = command.replace('/google/logs/action/done',
                                  shlex.quote(str(self.root)))
        self._run(command + '\ngcloud storage cp gs://bucket/in gs://bucket/out')


if __name__ == '__main__':
  unittest.main()
