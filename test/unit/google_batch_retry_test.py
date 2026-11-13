# Copyright 2026 Verily Life Sciences Inc. All Rights Reserved.
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
"""Exercise Batch retries through real GAPIC clients over mocked transports."""

import datetime
import unittest
from unittest import mock

from google.api_core import exceptions
from google.api_core import gapic_v1
from google.api_core import retry
from google.auth import credentials
from google.cloud import batch_v1
import pytz
import tenacity

from dsub.providers import google_batch
from test.unit import google_batch_test as batch_test_support


_TRANSIENT_ERRORS = (
    exceptions.DeadlineExceeded,
    exceptions.InternalServerError,
    exceptions.ServiceUnavailable,
    exceptions.TooManyRequests,
)


class BatchRetryTest(unittest.TestCase):

  def setUp(self):
    self.client = batch_v1.BatchServiceClient(
        credentials=credentials.AnonymousCredentials())
    self.addCleanup(self.client.transport.close)
    patcher = mock.patch.object(
        google_batch.batch_v1, 'BatchServiceClient', return_value=self.client)
    patcher.start()
    self.addCleanup(patcher.stop)
    sleep = mock.patch('time.sleep')
    sleep.start()
    self.addCleanup(sleep.stop)
    self.provider = google_batch.GoogleBatchJobProvider.__new__(
        google_batch.GoogleBatchJobProvider)
    self.provider._project = 'test-project'
    self.provider._location = 'us-central1'

  def _install_rpc(self, method, stub, default_retry=None):
    transport = self.client.transport
    rpc = getattr(transport, method)
    # Deliberately do not retry 504 by default. Tests must exercise the policy
    # the provider passes to GAPIC, including lazy requests for later pages.
    transport._wrapped_methods[rpc] = gapic_v1.method.wrap_method(
        stub,
        default_retry=(default_retry if default_retry is not None else
                       retry.Retry(predicate=lambda error: False)),
        default_timeout=60.0,
        client_info=gapic_v1.client_info.ClientInfo())

  def _list_case(self, method, page_size=7):
    if method == 'list_jobs':
      request = batch_v1.ListJobsRequest(
          parent='projects/p/locations/l', page_size=page_size)
      response = batch_v1.ListJobsResponse
      field = 'jobs'
      records = [batch_v1.Job(name='first'), batch_v1.Job(name='second')]
      fetch = self.provider._list_all_jobs
    else:
      request = batch_v1.ListTasksRequest(
          parent='projects/p/locations/l/jobs/j/taskGroups/group0',
          page_size=page_size)
      response = batch_v1.ListTasksResponse
      field = 'tasks'
      records = [batch_v1.Task(name='first'), batch_v1.Task(name='second')]
      fetch = self.provider._list_all_tasks
    return request, response, field, records, fetch

  def test_job_and_task_lists_retry_all_transient_errors_in_gapic(self):
    for method in ['list_jobs', 'list_tasks']:
      for error in _TRANSIENT_ERRORS:
        with self.subTest(method=method, error=error):
          request, response, field, records, fetch = self._list_case(method)
          stub = mock.Mock(side_effect=[
              error('transient failure'), response(**{field: records})])
          self._install_rpc(method, stub)
          # One outer attempt proves that GAPIC itself absorbs the failure.
          with mock.patch.object(fetch.retry, 'stop',
                                 tenacity.stop_after_attempt(1)):
            self.assertEqual(records, fetch(self.client, request))
          self.assertEqual(stub.call_count, 2)

  def test_job_and_task_lists_use_longer_rpc_deadline(self):
    for method in ['list_jobs', 'list_tasks']:
      with self.subTest(method=method):
        request, response, _, _, fetch = self._list_case(method)
        stub = mock.Mock(return_value=response())
        self._install_rpc(method, stub)
        self.assertEqual(fetch(self.client, request), [])
        self.assertAlmostEqual(
            stub.call_args.kwargs['timeout'],
            google_batch._BATCH_READ_TIMEOUT, delta=1.0)
        self.assertGreater(stub.call_args.kwargs['timeout'], 60.0)

  def test_job_and_task_lists_fail_fast_on_permanent_errors(self):
    for method in ['list_jobs', 'list_tasks']:
      for error in [exceptions.PermissionDenied, exceptions.InvalidArgument,
                    exceptions.NotFound]:
        with self.subTest(method=method, error=error):
          request, _, _, _, fetch = self._list_case(method)
          stub = mock.Mock(side_effect=error('permanent failure'))
          self._install_rpc(method, stub)
          with self.assertRaises(error):
            fetch(self.client, request)
          self.assertEqual(stub.call_count, 1)

  def test_later_page_failure_restarts_listing_without_duplicates(self):
    for method in ['list_jobs', 'list_tasks']:
      with self.subTest(method=method):
        request, response, field, records, fetch = self._list_case(method)
        requests = []
        outcomes = iter([
            response(**{field: records[:1]}, next_page_token='page2'),
            exceptions.DeadlineExceeded('later page failed'),
            response(**{field: records[:1]}, next_page_token='page2'),
            response(**{field: records[1:]}),
        ])

        def rpc(page_request, **unused_kwargs):
          requests.append(type(request)(page_request))
          outcome = next(outcomes)
          if isinstance(outcome, Exception):
            raise outcome
          return outcome

        self._install_rpc(method, mock.Mock(side_effect=rpc))
        with mock.patch.object(
            google_batch, '_BATCH_READ_RETRY',
            retry.Retry(predicate=lambda error: False)):
          self.assertEqual(fetch(self.client, request), records)
        self.assertEqual([r.page_token for r in requests],
                         ['', 'page2', '', 'page2'])
        self.assertEqual({r.page_size for r in requests}, {7})
        self.assertEqual(request.page_token, '')

  def test_outer_listing_retries_exhausted_inner_retry(self):
    for method in ['list_jobs', 'list_tasks']:
      with self.subTest(method=method):
        request, response, field, records, fetch = self._list_case(method)
        stub = mock.Mock(side_effect=[
            exceptions.RetryError('inner retry exhausted',
                                  exceptions.DeadlineExceeded('504')),
            response(**{field: records}),
        ])
        self._install_rpc(method, stub)
        self.assertEqual(fetch(self.client, request), records)
        self.assertEqual(stub.call_count, 2)

  def test_outer_listing_stops_after_five_attempts(self):
    for method in ['list_jobs', 'list_tasks']:
      for wrapped in [False, True]:
        with self.subTest(method=method, wrapped=wrapped):
          request, _, _, _, fetch = self._list_case(method)
          error = exceptions.DeadlineExceeded('504')
          if wrapped:
            error = exceptions.RetryError('inner retry exhausted', error)
          stub = mock.Mock(side_effect=error)
          self._install_rpc(method, stub)
          with mock.patch.object(
              google_batch, '_BATCH_READ_RETRY',
              retry.Retry(predicate=lambda error: False)):
            with self.assertRaises(type(error)):
              fetch(self.client, request)
          self.assertEqual(stub.call_count, google_batch._BATCH_LIST_ATTEMPTS)

  @staticmethod
  def _create_request():
    return batch_v1.CreateJobRequest(
        parent='projects/p/locations/l', job_id='j', job=batch_v1.Job())

  def test_submission_retries_transient_errors_with_fixed_job_id(self):
    for error in _TRANSIENT_ERRORS:
      with self.subTest(error=error):
        requests = []
        job = batch_v1.Job(name='projects/p/locations/l/jobs/j')

        def rpc(request, **unused_kwargs):
          requests.append(batch_v1.CreateJobRequest(request))
          if len(requests) == 1:
            raise error('transient failure')
          return job

        self._install_rpc('create_job', mock.Mock(side_effect=rpc))
        self.provider._submit_batch_job(self._create_request())
        self.assertEqual(len(requests), 2)
        self.assertEqual({r.job_id for r in requests}, {'j'})
        self.assertEqual({r.parent for r in requests},
                         {'projects/p/locations/l'})

  def test_ambiguous_create_fetches_existing_job_after_retry(self):
    job = batch_v1.Job(name='projects/p/locations/l/jobs/j')
    create = mock.Mock(side_effect=[
        exceptions.ServiceUnavailable('response lost'),
        exceptions.AlreadyExists('created by earlier attempt'),
    ])
    get = mock.Mock(side_effect=[exceptions.ServiceUnavailable('503'), job])
    self._install_rpc(
        'create_job', create,
        default_retry=retry.Retry(
            predicate=retry.if_exception_type(exceptions.ServiceUnavailable),
            initial=0, maximum=0, timeout=0.05))
    self._install_rpc('get_job', get)

    self.assertEqual(self.provider._create_job_with_retry(
        self.client, self._create_request()), job)

    self.assertEqual(create.call_count, 2)
    self.assertEqual(get.call_count, 2)
    self.assertEqual(get.call_args.args[0].name, job.name)
    self.assertAlmostEqual(get.call_args.kwargs['timeout'],
                           google_batch._BATCH_READ_TIMEOUT, delta=1.0)

  def test_first_attempt_duplicate_is_not_treated_as_our_job(self):
    create = mock.Mock(side_effect=exceptions.AlreadyExists('duplicate'))
    get = mock.Mock()
    self._install_rpc('create_job', create)
    self._install_rpc('get_job', get)
    with self.assertRaises(exceptions.AlreadyExists):
      self.provider._submit_batch_job(self._create_request())
    self.assertEqual(create.call_count, 1)
    get.assert_not_called()

  def test_create_fails_fast_on_permanent_errors(self):
    for error in [exceptions.PermissionDenied, exceptions.InvalidArgument]:
      with self.subTest(error=error):
        create = mock.Mock(side_effect=error('permanent failure'))
        self._install_rpc('create_job', create)
        with self.assertRaises(error):
          self.provider._submit_batch_job(self._create_request())
        self.assertEqual(create.call_count, 1)

  def test_create_attempt_limit_includes_every_rpc(self):
    create = mock.Mock(side_effect=exceptions.ServiceUnavailable('503'))
    # A GAPIC default would wrap the first error as RetryError. The provider
    # must explicitly disable it so every RPC belongs to the five-attempt cap.
    self._install_rpc(
        'create_job', create,
        default_retry=retry.Retry(
            predicate=retry.if_exception_type(exceptions.ServiceUnavailable),
            initial=0, maximum=0, timeout=0))
    with self.assertRaises(exceptions.ServiceUnavailable):
      self.provider._submit_batch_job(self._create_request())
    self.assertEqual(create.call_count, google_batch._CREATE_JOB_ATTEMPTS)

  def test_exact_job_lookup_retries_without_losing_missing_task_status(self):
    descriptor = batch_test_support.GoogleBatchProviderTest._job_descriptor(
        task_count=1)
    job, _ = self.provider._prepare_batch_request(descriptor)
    job = job.job
    job.name = 'projects/test-project/locations/us-central1/jobs/job-id'
    job.status.state = batch_v1.JobStatus.State.RUNNING
    job.create_time = datetime.datetime(2026, 1, 2, tzinfo=pytz.utc)
    self._install_rpc('list_jobs', mock.Mock(
        return_value=batch_v1.ListJobsResponse()))
    get = mock.Mock(side_effect=[exceptions.DeadlineExceeded('504'), job])
    self._install_rpc('get_job', get)
    self._install_rpc('list_tasks', mock.Mock(
        side_effect=exceptions.NotFound('TaskGroup not yet visible')))
    with mock.patch.object(self.provider, '_download_task_manifest',
                           return_value=None):
      tasks = list(self.provider.lookup_job_tasks(
          {'RUNNING'}, job_ids={'job-id'}))

    self.assertEqual(len(tasks), 1)
    self.assertEqual(tasks[0].get_field('job-id'), 'job-id')
    self.assertEqual(tasks[0].get_field('task-status'), 'RUNNING')
    self.assertIn('not yet visible', tasks[0].get_field('status-detail'))
    self.assertEqual(get.call_count, 2)
    self.assertAlmostEqual(get.call_args.kwargs['timeout'],
                           google_batch._BATCH_READ_TIMEOUT, delta=1.0)

  def test_transient_failures_are_visible_during_gapic_retries(self):
    request, response, _, _, fetch = self._list_case('list_jobs')
    self._install_rpc('list_jobs', mock.Mock(side_effect=[
        exceptions.DeadlineExceeded('visible transient failure'), response()]))
    with self.assertLogs(google_batch.__name__, level='WARNING') as logs:
      fetch(self.client, request)
    self.assertIn('retry', ' '.join(logs.output).lower())


if __name__ == '__main__':
  unittest.main()
