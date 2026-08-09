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
"""Unit tests for the Google Batch provider's Job-to-TaskGroup mapping."""

import datetime
import json
import subprocess
import unittest
from unittest import mock

from google.cloud import batch_v1
import pytz

from dsub.lib import job_model
from dsub.lib import param_util
from dsub.providers import google_batch


class GoogleBatchProviderTest(unittest.TestCase):

  def setUp(self):
    storage_patch = mock.patch.object(
        google_batch.dsub_util, 'get_storage_service'
    )
    self.addCleanup(storage_patch.stop)
    storage_patch.start()
    self.provider = google_batch.GoogleBatchJobProvider(
        dry_run=False, project='test-project', location='us-central1'
    )

  @staticmethod
  def _job_descriptor(task_count=3):
    logging = param_util.build_logging_param(
        'gs://test-bucket/logs/{job-id}/{task-id}.log'
    )
    job_resources = job_model.Resources(
        image='ubuntu:22.04',
        logging=logging,
        disk_size=job_model.DEFAULT_DISK_SIZE,
        disk_type=job_model.DEFAULT_DISK_TYPE,
        boot_disk_size=job_model.DEFAULT_BOOT_DISK_SIZE,
        regions=['us-central1'],
        zones=[],
        timeout='3600s',
        log_interval='60s',
        accelerator_count=0,
    )
    job_params = {
        'labels': {job_model.LabelParam('job-label', 'job-value')},
        'envs': set(),
        'inputs': set(),
        'outputs': set(),
        'input-recursives': set(),
        'output-recursives': set(),
        'mounts': set(),
    }
    task_descriptors = []
    for task_id in range(1, task_count + 1):
      task_params = {
          'labels': {
              job_model.LabelParam('task-label', f'task-{task_id}')
          },
          'envs': {
              job_model.EnvParam('SAMPLE_ID', f'sample-{task_id}'),
              # Verify internal runnable controls remain namespaced.
              job_model.EnvParam('INPUT_COUNT', 'user-value'),
          },
          'inputs': set(),
          'outputs': set(),
          'input-recursives': set(),
          'output-recursives': set(),
      }
      task_resources = job_model.Resources(
          preemptible=False,
          logging_path=param_util.build_logging_param(
              f'gs://test-bucket/logs/job-id/{task_id}.log'
          ),
      )
      task_descriptors.append(
          job_model.TaskDescriptor(
              {'task-id': task_id}, task_params, task_resources
          )
      )

    return job_model.JobDescriptor(
        {
            'job-id': 'job-id',
            'job-name': 'job-name',
            'user-id': 'user-id',
            'dsub-version': 'v0-5-1',
            'create-time': datetime.datetime(2026, 1, 2, tzinfo=pytz.utc),
            'script': job_model.Script('script.sh', '#!/bin/bash\necho ok'),
            'user-project': None,
            'task-ids': f'1-{task_count}',
        },
        job_params,
        job_resources,
        task_descriptors,
    )

  @staticmethod
  def _manifest_environments(manifest):
    environments = []
    for line in manifest.data.splitlines():
      environment = dict(manifest.common_environment)
      environment.update(json.loads(line))
      environments.append(environment)
    return environments

  def test_create_request_maps_one_job_to_multiple_batch_tasks(self):
    request, manifest = self.provider._prepare_batch_request(
        self._job_descriptor()
    )

    self.assertIsNotNone(manifest)
    self.assertEqual(request.job_id, 'job-id')
    self.assertEqual(len(request.job.task_groups), 1)
    task_group = request.job.task_groups[0]
    self.assertEqual(len(task_group.task_environments), 0)
    self.assertEqual(task_group.task_count, 3)
    self.assertEqual(task_group.task_count_per_node, 1)
    environments = self._manifest_environments(manifest)
    self.assertEqual(
        [environment['SAMPLE_ID'] for environment in environments],
        ['sample-1', 'sample-2', 'sample-3'],
    )
    first_environment = environments[0]
    self.assertEqual(first_environment['INPUT_COUNT'], 'user-value')
    self.assertEqual(
        first_environment['_DSUB_BATCH_LOCALIZATION_INPUT_COUNT'], '0'
    )
    self.assertNotIn('task-label', request.job.labels)

  def test_single_task_uses_task_count_and_inline_task_spec_environment(self):
    descriptor = self._job_descriptor(task_count=1)
    request, manifest = self.provider._prepare_batch_request(descriptor)

    self.assertIsNone(manifest)
    task_group = request.job.task_groups[0]
    self.assertEqual(task_group.task_count, 1)
    self.assertEqual(len(task_group.task_environments), 0)
    self.assertEqual(len(task_group.task_spec.runnables), 6)
    self.assertEqual(
        task_group.task_spec.environment.variables['SAMPLE_ID'], 'sample-1'
    )

    job = request.job
    job.name = 'projects/test-project/locations/us-central1/jobs/job-id'
    job.status = batch_v1.JobStatus(
        state=batch_v1.JobStatus.State.RUNNING
    )
    task = batch_v1.Task(
        name=f'{job.name}/taskGroups/group0/tasks/0',
        status=batch_v1.TaskStatus(
            state=batch_v1.TaskStatus.State.RUNNING
        ),
    )
    operation = google_batch.GoogleBatchOperation(job, task)
    self.assertEqual(operation.get_field('task-id'), '1')
    self.assertEqual(operation.get_field('envs')['SAMPLE_ID'], 'sample-1')

  def test_multi_task_request_shape_has_no_200_task_boundary(self):
    for task_count in [2, 200, 201]:
      with self.subTest(task_count=task_count):
        request, manifest = self.provider._prepare_batch_request(
            self._job_descriptor(task_count=task_count)
        )
        task_group = request.job.task_groups[0]
        self.assertIsNotNone(manifest)
        self.assertEqual(task_group.task_count, task_count)
        self.assertEqual(len(task_group.task_environments), 0)
        self.assertEqual(len(task_group.task_spec.runnables), 7)

  def test_submit_job_creates_one_batch_job(self):
    client = mock.Mock()
    client.create_job.return_value = batch_v1.Job(
        name='projects/test-project/locations/us-central1/jobs/job-id'
    )
    with mock.patch.object(
        google_batch.batch_v1, 'BatchServiceClient', return_value=client
    ):
      result = self.provider.submit_job(
          self._job_descriptor(), skip_if_output_present=False
      )

    client.create_job.assert_called_once()
    request = client.create_job.call_args.kwargs['request']
    self.assertEqual(request.job.task_groups[0].task_count, 3)
    self.assertEqual(len(request.job.task_groups[0].task_environments), 0)
    self.assertEqual(result['task-id'], [1, 2, 3])

  def test_submit_job_omits_skipped_tasks_from_the_one_batch_job(self):
    client = mock.Mock()
    client.create_job.return_value = batch_v1.Job(
        name='projects/test-project/locations/us-central1/jobs/job-id'
    )
    with mock.patch.object(
        google_batch.batch_v1, 'BatchServiceClient', return_value=client
    ), mock.patch.object(
        google_batch.dsub_util,
        'outputs_are_present',
        side_effect=[False, True, False],
    ), mock.patch.object(
        self.provider, '_stage_task_manifest'
    ) as stage_manifest:
      result = self.provider.submit_job(
          self._job_descriptor(), skip_if_output_present=True
      )

    client.create_job.assert_called_once()
    request = client.create_job.call_args.kwargs['request']
    self.assertEqual(request.job.task_groups[0].task_count, 2)
    manifest = stage_manifest.call_args.args[0]
    environments = self._manifest_environments(manifest)
    self.assertEqual(
        [
            environment['SAMPLE_ID']
            for environment in environments
        ],
        ['sample-1', 'sample-3'],
    )
    self.assertEqual(result['task-id'], [1, 3])

  def test_operation_reads_metadata_and_status_for_one_batch_task(self):
    request, manifest = self.provider._prepare_batch_request(
        self._job_descriptor()
    )
    environments = self._manifest_environments(manifest)
    job = request.job
    job.name = 'projects/test-project/locations/us-central1/jobs/job-id'
    job.status = batch_v1.JobStatus(
        state=batch_v1.JobStatus.State.FAILED
    )
    task = batch_v1.Task(
        name=f'{job.name}/taskGroups/group0/tasks/1',
        status=batch_v1.TaskStatus(
            state=batch_v1.TaskStatus.State.SUCCEEDED
        ),
    )

    operation = google_batch.GoogleBatchOperation(job, task, environments[1])

    self.assertEqual(operation.get_field('job-id'), 'job-id')
    self.assertEqual(operation.get_field('task-id'), '2')
    self.assertEqual(operation.get_field('task-status'), 'SUCCESS')
    self.assertEqual(operation.get_field('envs')['SAMPLE_ID'], 'sample-2')
    self.assertEqual(
        operation.get_field('labels')['task-label'], 'task-2'
    )

  def test_unexecuted_task_tracks_job_cancellation(self):
    request, manifest = self.provider._prepare_batch_request(
        self._job_descriptor()
    )
    job = request.job
    job.name = 'projects/test-project/locations/us-central1/jobs/job-id'
    job.status = batch_v1.JobStatus(
        state=batch_v1.JobStatus.State.CANCELLATION_IN_PROGRESS
    )
    task = batch_v1.Task(
        name=f'{job.name}/taskGroups/group0/tasks/0',
        status=batch_v1.TaskStatus(
            state=batch_v1.TaskStatus.State.UNEXECUTED
        ),
    )
    operation = google_batch.GoogleBatchOperation(
        job, task, self._manifest_environments(manifest)[0]
    )

    self.assertEqual(operation.get_field('task-status'), 'RUNNING')
    job.status.state = batch_v1.JobStatus.State.CANCELLED
    self.assertEqual(operation.get_field('task-status'), 'CANCELED')

  def test_delete_cancels_one_provider_job_for_all_its_tasks(self):
    first_task = mock.Mock()
    second_task = mock.Mock()
    first_task.get_field.return_value = 'provider-job'
    second_task.get_field.return_value = 'provider-job'
    client = mock.Mock()

    with mock.patch.object(
        self.provider,
        'lookup_job_tasks',
        return_value=iter([first_task, second_task]),
    ), mock.patch.object(
        google_batch.batch_v1, 'BatchServiceClient', return_value=client
    ), mock.patch.object(
        google_batch.google_base,
        'cancel',
        return_value=([first_task], []),
    ) as cancel:
      canceled_tasks, errors = self.provider.delete_jobs(
          user_ids={'user-id'},
          job_ids={'job-id'},
          task_ids={'*'},
          labels=None,
      )

    self.assertEqual(canceled_tasks, [first_task, second_task])
    self.assertEqual(errors, [])
    self.assertEqual(cancel.call_args.args[2], [first_task])

  def test_lookup_expands_jobs_and_filters_at_task_level(self):
    request, manifest = self.provider._prepare_batch_request(
        self._job_descriptor()
    )
    environments = self._manifest_environments(manifest)
    job = request.job
    job.name = 'projects/test-project/locations/us-central1/jobs/job-id'
    job.status = batch_v1.JobStatus(
        state=batch_v1.JobStatus.State.FAILED
    )
    job.create_time = datetime.datetime(
        2026, 1, 2, 3, 4, 5, 123000, tzinfo=pytz.utc
    )
    job.update_time = datetime.datetime(
        2026, 1, 2, 3, 5, 5, 123000, tzinfo=pytz.utc
    )
    batch_tasks = [
        batch_v1.Task(
            name=f'{job.name}/taskGroups/group0/tasks/{index}',
            status=batch_v1.TaskStatus(state=state),
        )
        for index, state in enumerate(
            [
                batch_v1.TaskStatus.State.FAILED,
                batch_v1.TaskStatus.State.SUCCEEDED,
                batch_v1.TaskStatus.State.FAILED,
            ]
        )
    ]
    client = mock.Mock()
    client.list_jobs.return_value = [job]
    client.list_tasks.return_value = batch_tasks

    with mock.patch.object(
        google_batch.batch_v1, 'BatchServiceClient', return_value=client
    ), mock.patch.object(
        self.provider,
        '_download_task_manifest',
        return_value=environments,
    ):
      tasks = list(
          self.provider.lookup_job_tasks(
              {'SUCCESS'}, job_ids={'job-id'}, task_ids={'2'}
          )
      )

    self.assertEqual([task.get_field('task-id') for task in tasks], ['2'])
    list_jobs_request = client.list_jobs.call_args.kwargs['request']
    self.assertNotIn('status.state', list_jobs_request.filter)
    self.assertNotIn('task-id', list_jobs_request.filter)
    list_tasks_request = client.list_tasks.call_args.kwargs['request']
    self.assertEqual(
        list_tasks_request.parent, f'{job.name}/taskGroups/group0'
    )

  def test_large_task_array_uses_indexed_manifest(self):
    descriptor = self._job_descriptor(task_count=201)
    manifest = self.provider._build_task_manifest(descriptor)
    request = self.provider._create_batch_request(descriptor, manifest)

    task_group = request.job.task_groups[0]
    self.assertEqual(task_group.task_count, 201)
    self.assertEqual(len(task_group.task_environments), 0)
    self.assertEqual(len(task_group.task_spec.runnables), 7)
    self.assertEqual(
        task_group.task_spec.environment.variables[
            google_batch._TASK_MANIFEST_URI_VARNAME
        ],
        manifest.uri,
    )

    environments = self._manifest_environments(manifest)
    self.assertEqual(len(environments), 201)
    self.assertEqual(environments[200]['SAMPLE_ID'], 'sample-201')
    index_record_size = google_batch._TASK_MANIFEST_INDEX_RECORD_SIZE
    index_record = manifest.index[
        200 * index_record_size : 201 * index_record_size
    ].decode('ascii')
    start, end = [int(value) for value in index_record.split()]
    self.assertEqual(
        json.loads(manifest.data[start : end + 1])['SAMPLE_ID'],
        'sample-201',
    )
    subprocess.run(
        ['bash', '-n', '-c', google_batch._TASK_MANIFEST_BOOTSTRAP_CMD],
        check=True,
    )

  def test_manifest_staging_and_download_preserve_complete_environments(self):
    descriptor = self._job_descriptor(task_count=201)
    manifest = self.provider._build_task_manifest(descriptor)
    request = self.provider._create_batch_request(descriptor, manifest)

    with mock.patch.object(
        self.provider, '_upload_gcs_object'
    ) as upload:
      self.provider._stage_task_manifest(manifest)

    self.assertEqual(upload.call_args_list[0].args[0], manifest.uri)
    self.assertEqual(upload.call_args_list[1].args[0], manifest.index_uri)

    with mock.patch.object(
        self.provider, '_download_gcs_object', return_value=manifest.data
    ):
      environments = self.provider._download_task_manifest(request.job)

    self.assertEqual(len(environments), 201)
    self.assertEqual(environments[200]['SAMPLE_ID'], 'sample-201')
    self.assertIn(
        '_DSUB_BATCH_PREPARE__SCRIPT_REPR', environments[200]
    )

  def test_submit_large_job_stages_manifest_and_creates_one_job(self):
    client = mock.Mock()
    client.create_job.return_value = batch_v1.Job(
        name='projects/test-project/locations/us-central1/jobs/job-id'
    )
    with mock.patch.object(
        google_batch.batch_v1, 'BatchServiceClient', return_value=client
    ), mock.patch.object(
        self.provider, '_stage_task_manifest'
    ) as stage_manifest:
      result = self.provider.submit_job(
          self._job_descriptor(task_count=201),
          skip_if_output_present=False,
      )

    client.create_job.assert_called_once()
    stage_manifest.assert_called_once()
    request = client.create_job.call_args.kwargs['request']
    self.assertEqual(request.job.task_groups[0].task_count, 201)
    self.assertEqual(len(result['task-id']), 201)

  def test_lookup_uses_manifest_to_recover_large_job_task_metadata(self):
    descriptor = self._job_descriptor(task_count=201)
    manifest = self.provider._build_task_manifest(descriptor)
    environments = self._manifest_environments(manifest)
    request = self.provider._create_batch_request(descriptor, manifest)
    job = request.job
    job.name = 'projects/test-project/locations/us-central1/jobs/job-id'
    job.status = batch_v1.JobStatus(
        state=batch_v1.JobStatus.State.RUNNING
    )
    job.create_time = datetime.datetime(
        2026, 1, 2, 3, 4, 5, 123000, tzinfo=pytz.utc
    )
    job.update_time = datetime.datetime(
        2026, 1, 2, 3, 5, 5, 123000, tzinfo=pytz.utc
    )
    batch_task = batch_v1.Task(
        name=f'{job.name}/taskGroups/group0/tasks/200',
        status=batch_v1.TaskStatus(
            state=batch_v1.TaskStatus.State.RUNNING
        ),
    )
    client = mock.Mock()
    client.list_jobs.return_value = [job]
    client.list_tasks.return_value = [batch_task]

    with mock.patch.object(
        google_batch.batch_v1, 'BatchServiceClient', return_value=client
    ), mock.patch.object(
        self.provider,
        '_download_task_manifest',
        return_value=environments,
    ):
      tasks = list(
          self.provider.lookup_job_tasks(
              {'RUNNING'}, job_ids={'job-id'}, task_ids={'201'}
          )
      )

    self.assertEqual(len(tasks), 1)
    self.assertEqual(tasks[0].get_field('task-id'), '201')
    self.assertEqual(tasks[0].get_field('envs')['SAMPLE_ID'], 'sample-201')

  def test_rejects_jobs_over_native_batch_task_limit(self):
    with mock.patch.object(google_batch, '_MAX_TASKS_PER_JOB', 2), \
        self.assertRaisesRegex(ValueError, 'at most 2 tasks'):
      self.provider.submit_job(
          self._job_descriptor(task_count=3),
          skip_if_output_present=False,
      )


if __name__ == '__main__':
  unittest.main()
