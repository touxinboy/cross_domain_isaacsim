"""Restrict file-content writes without user namespaces, then exec a Python app.

Linux Landlock ABI >= 3 is required. Reads and device ioctls remain available;
file creation/removal/content writes are allowed under ground and /dev/shm.
Device files and procfs metadata retain ordinary user write access. This is
not mount/IPC isolation or protection against arbitrary metadata operations.
See https://docs.kernel.org/userspace-api/landlock.html.
"""

import argparse
import ctypes
import os
from pathlib import Path
import platform
import runpy
import sys


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def restrict_writes(root):
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("Landlock syscall numbers are only configured for x86_64/aarch64.")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long

    def call(number, *args):
        result = libc.syscall(ctypes.c_long(number), *args)
        if result < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return result

    abi = call(444, ctypes.c_void_p(), ctypes.c_size_t(0), ctypes.c_uint(1))
    if abi < 3:
        raise RuntimeError(f"Landlock ABI >= 3 required for truncate protection; found {abi}.")
    # WRITE_FILE, REMOVE_*, MAKE_*, REFER and TRUNCATE; reads/exec are unrestricted.
    write_file = 1 << 1
    writes = write_file | sum(1 << bit for bit in range(4, 15))
    attr = Ruleset(writes)
    ruleset = call(444, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint(0))
    try:
        # CUDA names threads through procfs. Dynamic /proc/self/task dentries
        # are unsuitable as stable Landlock hierarchy roots. Preserve ordinary
        # DAC permissions on procfs writes; these are not persistent files.
        for directory, rights in ((root, writes), (Path('/dev'), write_file),
                                  (Path('/dev/shm'), writes),
                                  (Path('/proc'), write_file | (1 << 14))):
            fd = os.open(directory, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(rights, fd)
                call(445, ctypes.c_int(ruleset), ctypes.c_int(1), ctypes.byref(rule), ctypes.c_uint(0))
            finally:
                os.close(fd)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        call(446, ctypes.c_int(ruleset), ctypes.c_uint(0))
    finally:
        os.close(ruleset)
    return abi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--applications', type=Path)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    applications = args.applications.resolve(strict=True) if args.applications else None
    if applications and not applications.is_relative_to(root):
        parser.error('Applications directory must be inside the writable root.')
    abi = restrict_writes(root)
    if args.probe:
        print(f'Landlock ABI {abi}: file write restrictions active', flush=True)
        return 0
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A Python entrypoint is required after --.')
    if Path(command[0]).resolve() == Path(sys.executable).resolve():
        os.execv(command[0], command)
    # Kit 4.5 hardcodes its Python desktop shortcut path instead of respecting
    # XDG_DATA_HOME. Redirect just those two expanduser calls, not HOME itself.
    if applications:
        original_expanduser = os.path.expanduser
        redirects = {'~/.local/share/applications': str(applications),
                     '~/.local/share/applications/IsaacSim.desktop': str(applications / 'IsaacSim.desktop')}

        def expanduser(path):
            return redirects.get(os.fspath(path), original_expanduser(path))

        os.path.expanduser = expanduser
    sys.argv = command
    runpy.run_path(command[0], run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
