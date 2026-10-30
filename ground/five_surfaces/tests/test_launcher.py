import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('ground_launch', SCRIPTS / 'launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LauncherTests(unittest.TestCase):
    def test_denied_uid_map_selects_landlock(self):
        denial = subprocess.CompletedProcess([], 1, '', 'bwrap: setting up uid map: Permission denied')
        supported = subprocess.CompletedProcess([], 0, 'Landlock ABI 8', '')
        with patch.object(launch.Path, 'is_file', return_value=True), patch.object(launch.subprocess, 'run', side_effect=[denial, supported]):
            backend, reason = launch.select_isolation('auto', SCRIPTS.parent.parent)
        self.assertEqual(backend, 'landlock')
        self.assertIn('uid map', reason)

    def test_no_unisolated_launch_when_both_backends_fail(self):
        failure = subprocess.CompletedProcess([], 1, '', 'Permission denied')
        with patch.object(launch.Path, 'is_file', return_value=True), patch.object(launch.subprocess, 'run', return_value=failure):
            with self.assertRaisesRegex(RuntimeError, 'No supported file write isolation'):
                launch.select_isolation('auto', SCRIPTS.parent.parent)

    def test_explicit_bwrap_reports_denial(self):
        failure = subprocess.CompletedProcess([], 1, '', 'uid map: Permission denied')
        with patch.object(launch.Path, 'is_file', return_value=True), patch.object(launch.subprocess, 'run', return_value=failure):
            with self.assertRaisesRegex(RuntimeError, 'bwrap is unavailable'):
                launch.select_isolation('bwrap', SCRIPTS.parent.parent)

    def test_landlock_preserves_reads_blocks_mutation_and_is_inherited(self):
        with tempfile.TemporaryDirectory(prefix='ground_landlock_test_') as directory:
            fixture = Path(directory)
            root = fixture / 'allowed'
            root.mkdir()
            applications = root / 'applications'
            applications.mkdir()
            outside = fixture / 'outside.txt'
            outside.write_text('preserve')
            runner = root / 'probe.py'
            runner.write_text(r'''
import errno, json, os, pathlib, subprocess, sys
root, outside = map(pathlib.Path, sys.argv[1:])
assert outside.read_text() == 'preserve'
# CUDA must be allowed to name threads through this volatile procfs file.
fd = os.open(f'/proc/self/task/{os.getpid()}/comm', os.O_WRONLY | os.O_TRUNC)
os.write(fd, b'ground-probe')
os.close(fd)
own = root / 'own.txt'
own.write_text('allowed')
own.rename(root / 'renamed.txt')
(root / 'renamed.txt').unlink()
attempts = [lambda: outside.write_text('bad'), lambda: os.truncate(outside, 0),
            lambda: outside.unlink(), lambda: (outside.parent / 'new.txt').write_text('bad'),
            lambda: outside.rename(root / 'stolen.txt')]
for operation in attempts:
    try: operation()
    except OSError as error: assert error.errno in (errno.EACCES, errno.EPERM, errno.EROFS)
    else: raise AssertionError('outside mutation was allowed')
child_code = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('bad')"
child = subprocess.run([sys.executable, '-B', '-c', child_code, str(outside)], capture_output=True)
assert child.returncode != 0, 'child escaped restrictions'
shortcut = pathlib.Path(os.path.expanduser('~/.local/share/applications/IsaacSim.desktop'))
assert shortcut == root / 'applications/IsaacSim.desktop'
shortcut.write_text('own shortcut')
print(json.dumps({'operations_blocked': len(attempts), 'child_blocked': True, 'shortcut_redirected': True}))
''')
            result = subprocess.run([sys.executable, '-B', str(SCRIPTS / 'landlock_exec.py'),
                                     '--root', str(root), '--applications', str(applications), '--',
                                     str(runner), str(root), str(outside)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['operations_blocked'], 5)
            self.assertEqual(outside.read_text(), 'preserve')
            self.assertFalse((fixture / 'new.txt').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
