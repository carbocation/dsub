# Copyright 2018 Verily Life Sciences Inc. All Rights Reserved.
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
"""Unit tests for the local provider.
"""

import copy
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from dsub.lib import dsub_util
from dsub.lib import job_model
from dsub.lib import param_util
from dsub.providers import local
import pytz

# The local provider can launch tasks quickly enough that they end up with the
# same timestamp. The goal of this test below is to verify that tasks come back
# sorted in a way expected by users: "most recent first".
#
# Also - originally the local provider did not serialize out the task create
# time (just the job create time), so we need to be sure the sort routine does
# not blow up if one is not provided.


# Define a utility function for creating a task object with minimal elements
# required for the sort test.
def _make_task(job_create_time, task_create_time, task_id):
  return local.LocalTask(
      job_descriptor=job_model.JobDescriptor(
          job_metadata={'create-time': job_create_time},
          job_params=None,
          job_resources=None,
          task_descriptors=[
              job_model.TaskDescriptor(
                  task_metadata={
                      'task-id': task_id,
                      'create-time': task_create_time
                  },
                  task_params=None,
                  task_resources=None)
          ]),
      task_status=None,
      log_detail=None,
      end_time=None,
      last_update=None,
      pid=None,
      events=None,
      script=None)


CREATE_TIME_1 = dsub_util.replace_timezone(
    datetime.datetime(2018, 1, 1, 22, 28, 37, 321788), pytz.utc)

CREATE_TIME_2 = dsub_util.replace_timezone(
    datetime.datetime(2018, 1, 2, 22, 28, 37, 321788), pytz.utc)

CREATE_TIME_3 = dsub_util.replace_timezone(
    datetime.datetime(2018, 1, 3, 22, 28, 37, 321788), pytz.utc)

# Simple list of tasks that are in exactly the wrong order:
#  Job 1: two tasks (both with timestamp 1)
#  Job 2: two tasks  (both with timestamp 2)
#  Job 3: one implicit task (with timestamp 3)
# To ensure that the sort picks up the task create time, but falls back to the
# job create time, set the job create time the same for all of them.
REVERSE_ORDERED_TASKS = [
    _make_task(CREATE_TIME_1, None, 1),
    _make_task(CREATE_TIME_1, None, 2),
    _make_task(CREATE_TIME_1, CREATE_TIME_2, 1),
    _make_task(CREATE_TIME_1, CREATE_TIME_2, 2),
    _make_task(CREATE_TIME_1, CREATE_TIME_3, None),
]


class TestSortTasks(unittest.TestCase):

  def test_sort_tasks(self):
    tasks = copy.deepcopy(REVERSE_ORDERED_TASKS)
    local._sort_tasks(tasks)

    self.assertEqual(len(tasks), len(REVERSE_ORDERED_TASKS))
    self.assertEqual(tasks[0].get_field('create-time'), CREATE_TIME_3)
    self.assertEqual(tasks[0].get_field('task-id'), None)
    self.assertEqual(tasks[1].get_field('create-time'), CREATE_TIME_2)
    self.assertEqual(tasks[1].get_field('task-id'), 2)
    self.assertEqual(tasks[2].get_field('create-time'), CREATE_TIME_2)
    self.assertEqual(tasks[2].get_field('task-id'), 1)
    self.assertEqual(tasks[3].get_field('create-time'), CREATE_TIME_1)
    self.assertEqual(tasks[3].get_field('task-id'), 2)
    self.assertEqual(tasks[4].get_field('create-time'), CREATE_TIME_1)
    self.assertEqual(tasks[4].get_field('task-id'), 1)


@unittest.skipUnless(shutil.which('rsync'), 'rsync is required for local copies')
class TestLocalFileCopies(unittest.TestCase):
  """Execute local staging commands without a cloud CLI or Docker."""

  def setUp(self):
    directory = tempfile.TemporaryDirectory(prefix='dsub local copies ')
    self.addCleanup(directory.cleanup)
    self.root = Path(directory.name)
    self.task_dir = self.root / 'task directory'
    self.provider = local.LocalJobProvider(resources=None)
    bin_dir = self.root / 'bin'
    bin_dir.mkdir()
    for command in ['rsync', 'mkdir']:
      (bin_dir / command).symlink_to(shutil.which(command))
    for command in ['gcloud', 'gsutil']:
      executable = bin_dir / command
      executable.write_text(
          '#!/bin/bash\necho "Cloud CLI used for a local copy" >&2\nexit 99\n')
      executable.chmod(0o755)
    self.environment = dict(os.environ, PATH=str(bin_dir))

  def _run(self, command):
    result = subprocess.run(
        ['/bin/bash', '-e', '-c', command], env=self.environment,
        capture_output=True, text=True, timeout=10)
    self.assertEqual(result.returncode, 0, result.stderr)

  def _write(self, path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)

  def _staged_path(self, param):
    return self.task_dir / 'data' / param.docker_path

  def test_localize_single_file_with_spaces(self):
    source = self.root / 'source directory' / 'input file.txt'
    self._write(source, 'single input\n')
    param = param_util.InputFileParamUtil('input').make_param(
        'INPUT', str(source), recursive=False)

    self._run(self.provider._localize_inputs_command(
        str(self.task_dir), [param], user_project='billing-project'))

    self.assertEqual(self._staged_path(param).read_text(), 'single input\n')

  def test_localize_follows_file_symlink(self):
    target = self.root / 'link targets' / 'input target.txt'
    self._write(target, 'linked input\n')
    source = self.root / 'input link.txt'
    source.symlink_to(target)
    param = param_util.InputFileParamUtil('input').make_param(
        'INPUT', str(source), recursive=False)

    self._run(self.provider._localize_inputs_command(
        str(self.task_dir), [param], user_project=None))

    staged = self._staged_path(param)
    self.assertTrue(staged.is_file())
    self.assertFalse(staged.is_symlink())
    self.assertEqual(staged.read_text(), 'linked input\n')

  def test_localize_wildcards_copy_files_and_skip_directories(self):
    for count in [1, 2]:
      with self.subTest(matches=count):
        source_dir = self.root / ('source files %s' % count)
        expected = {}
        for index in range(count):
          name = 'input file %s.txt' % index
          expected[name] = 'input %s\n' % index
          self._write(source_dir / name, expected[name])
        self._write(source_dir / 'input directory.txt' / 'child.txt', 'skip\n')
        (source_dir / 'input linked directory.txt').symlink_to(
            source_dir / 'input directory.txt', target_is_directory=True)
        param = param_util.InputFileParamUtil('input').make_param(
            'INPUT', str(source_dir / 'input *.txt'), recursive=False)

        self._run(self.provider._localize_inputs_command(
            str(self.task_dir), [param], user_project=None))

        destination = self._staged_path(param).parent
        self.assertEqual(
            {path.name: path.read_text() for path in destination.iterdir()},
            expected)

  def test_delocalize_single_file_with_spaces(self):
    destination = self.root / 'output directory' / 'output file.txt'
    param = param_util.OutputFileParamUtil('output').make_param(
        'OUTPUT', str(destination), recursive=False)
    self._write(self._staged_path(param), 'single output\n')

    self._run(self.provider._delocalize_outputs_commands(
        str(self.task_dir), [param], user_project='billing-project'))

    self.assertEqual(destination.read_text(), 'single output\n')

  def test_delocalize_follows_file_symlink(self):
    target = self.root / 'link targets' / 'output target.txt'
    self._write(target, 'linked output\n')
    destination = self.root / 'output directory' / 'output link.txt'
    param = param_util.OutputFileParamUtil('output').make_param(
        'OUTPUT', str(destination), recursive=False)
    staged = self._staged_path(param)
    staged.parent.mkdir(parents=True)
    staged.symlink_to(target)

    self._run(self.provider._delocalize_outputs_commands(
        str(self.task_dir), [param], user_project=None))

    self.assertTrue(destination.is_file())
    self.assertFalse(destination.is_symlink())
    self.assertEqual(destination.read_text(), 'linked output\n')

  def test_delocalize_wildcards_copy_files_and_skip_directories(self):
    for count in [1, 2]:
      with self.subTest(matches=count):
        destination = self.root / ('output files %s' % count)
        param = param_util.OutputFileParamUtil('output').make_param(
            'OUTPUT', str(destination / 'output *.txt'), recursive=False)
        source_dir = self._staged_path(param).parent
        expected = {}
        for index in range(count):
          name = 'output file %s.txt' % index
          expected[name] = 'output %s\n' % index
          self._write(source_dir / name, expected[name])
        self._write(source_dir / 'output directory.txt' / 'child.txt', 'skip\n')
        (source_dir / 'output linked directory.txt').symlink_to(
            source_dir / 'output directory.txt', target_is_directory=True)

        self._run(self.provider._delocalize_outputs_commands(
            str(self.task_dir), [param], user_project=None))

        self.assertEqual(
            {path.name: path.read_text() for path in destination.iterdir()},
            expected)


if __name__ == '__main__':
  unittest.main()
