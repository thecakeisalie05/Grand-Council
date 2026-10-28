"""Download and manage official llama.cpp binaries without user Python/CUDA toolchains.

Windows x64 desktop targets are first-class. Other OSes may use an existing
llama-server on PATH, an OpenAI-compatible endpoint, or legacy Python adapters.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import httpx

RELEASES_URL = 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=6'
USER_AGENT = 'Grand-Council/0.2.0 (https://github.com/thecakeisalie05/Grand-Council)'
GIB = 1024 ** 3


def _command(args, timeout=8):
    try:
        options = {'capture_output': True, 'text': True, 'timeout': timeout,
                   'encoding': 'utf-8', 'errors': 'replace'}
        if os.name == 'nt':
            options['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        done = subprocess.run(args, **options)
        return done.stdout.strip() if done.returncode == 0 else ''
    except (OSError, subprocess.TimeoutExpired):
        return ''


def hardware_profile():
    """Probe GPU drivers independently of torch (which is not bundled)."""
    result = {'platform': platform.system(), 'machine': platform.machine(),
              'gpu': [], 'preferred': ['cpu'], 'reason': 'CPU fallback is always available.'}
    if platform.system() == 'Windows':
        powershell = _command(['powershell', '-NoProfile', '-NonInteractive', '-Command',
            'Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress'])
        if powershell:
            try:
                devices = json.loads(powershell)
                for d in devices if isinstance(devices, list) else [devices]:
                    if d.get('Name'):
                        result['gpu'].append({'name': d['Name'], 'vram_gb': None, 'source': 'Windows'})
            except (ValueError, TypeError):
                pass
    nvidia = _command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                       '--format=csv,noheader,nounits'])
    if nvidia and platform.system() == 'Windows':
        for line in nvidia.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 3:
                continue
            try:
                gb = round(float(parts[1]) / 1024, 1)
            except ValueError:
                gb = None
            entry = {'name': parts[0], 'vram_gb': gb, 'driver': parts[2], 'source': 'NVIDIA'}
            result['gpu'] = [x for x in result['gpu'] if x['name'].lower() != entry['name'].lower()]
            result['gpu'].insert(0, entry)
        # CUDA 12.4's Windows driver baseline is 551.61. The latest CUDA 13
        # drops support for Pascal, so use CUDA 12 even on modern drivers.
        try:
            driver = int(parts[2].split('.')[0])
        except (ValueError, IndexError):
            driver = 0
        if driver >= 552:
            result['preferred'] = ['cuda12', 'vulkan', 'cpu']
            result['reason'] = 'NVIDIA GPU and CUDA 12-capable driver detected.'
        else:
            result['preferred'] = ['vulkan', 'cpu']
            result['reason'] = 'NVIDIA driver is too old or unknown for bundled CUDA 12; trying Vulkan.'
    elif result['gpu']:
        names = ' '.join(x['name'].lower() for x in result['gpu'])
        if any(name in names for name in ('amd', 'radeon', 'intel', 'arc', 'nvidia', 'geforce')):
            result['preferred'] = ['vulkan', 'cpu']
            result['reason'] = 'Detected a display GPU; probing Vulkan at model startup.'
    return result


def _get_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def select_release(releases, backend):
    """Select matching official assets. GitHub `latest` can be a tag-only release."""
    suffix = {'cuda12': '-bin-win-cuda-12.4-x64.zip',
              'vulkan': '-bin-win-vulkan-x64.zip',
              'cpu': '-bin-win-cpu-x64.zip'}[backend]
    for release in releases:
        if release.get('draft') or release.get('prerelease'):
            continue
        assets = release.get('assets') or []
        executables = [a for a in assets if a.get('name', '').startswith('llama-')
                       and a['name'].endswith(suffix)]
        if not executables:
            continue
        wanted = executables[:1]
        if backend == 'cuda12':
            libs = [a for a in assets if a.get('name') == 'cudart-llama-bin-win-cuda-12.4-x64.zip']
            if not libs:
                continue
            wanted += libs[:1]
        return release['tag_name'], wanted
    raise RuntimeError(f'Official llama.cpp releases do not have a compatible Windows {backend} binary.')


def _safe_extract(archive_path, destination):
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            path = Path(item.filename.replace('\\', '/'))
            if path.is_absolute() or '..' in path.parts or ':' in item.filename or item.file_size > 2 * GIB:
                raise RuntimeError('Unsafe archive member in runtime download.')
        archive.extractall(destination)


def _download_verified(asset, destination):
    digest = asset.get('digest') or ''
    if not digest.startswith('sha256:'):
        raise RuntimeError(f'Official release asset {asset["name"]} has no SHA256 digest; refusing unverified binary.')
    url = asset.get('browser_download_url', '')
    if not url.startswith('https://github.com/ggml-org/llama.cpp/releases/download/'):
        raise RuntimeError('Unexpected llama.cpp binary download location.')
    checksum = hashlib.sha256()
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response, open(destination, 'wb') as out:
        while chunk := response.read(1024 * 1024):
            checksum.update(chunk)
            out.write(chunk)
    if checksum.hexdigest().lower() != digest.partition(':')[2].lower():
        raise RuntimeError('llama.cpp archive SHA256 mismatch; refusing to execute download.')


class RuntimeManager:
    def __init__(self, home):
        self.home = Path(home)
        self.lock = threading.RLock()
        self.processes = []

    def _location(self, backend):
        return self.home / 'runtimes' / 'llama.cpp' / backend

    def installed(self):
        return {backend: self.find_exe(backend) is not None for backend in ('cuda12', 'vulkan', 'cpu')}

    def find_exe(self, backend):
        directory = self._location(backend)
        if not directory.exists():
            return None
        matches = sorted(directory.rglob('llama-server.exe'))
        return matches[0] if matches else None

    def install(self, backend, progress=None, releases=None):
        if backend not in ('cuda12', 'vulkan', 'cpu'):
            raise ValueError('Supported Windows backends: cuda12, vulkan, cpu.')
        if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64', 'x86_64'):
            raise RuntimeError('Automatic binaries currently target Windows x64. Use an existing llama-server on other systems.')
        with self.lock:
            found = self.find_exe(backend)
            if found:
                return found
            progress = progress or (lambda message: None)
            progress(f'Finding official llama.cpp {backend} release…')
            tag, assets = select_release(releases if releases is not None else _get_json(RELEASES_URL), backend)
            expected = sum(int(a.get('size', 0)) for a in assets)
            disk = shutil.disk_usage(self.home)
            # downloads + expanded archives. Conservative safety floor, not an exact estimate.
            minimum = max(0.5 * GIB, expected * 2.2)
            if disk.free < minimum:
                raise RuntimeError(f'Insufficient space for {backend} backend: need ~{minimum / GIB:.1f} GB '
                                   f'free, have {disk.free / GIB:.1f} GB. Try Vulkan (~small download) or CPU.')
            destination = self._location(backend)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='gc-llama-', dir=str(destination.parent)) as scratch:
                temp = Path(scratch)
                stage = temp / 'staging'
                stage.mkdir()
                for asset in assets:
                    progress(f'Downloading and verifying {asset["name"]} ({asset["size"] // 1048576} MB)…')
                    archive = temp / asset['name']
                    _download_verified(asset, archive)
                    _safe_extract(archive, stage)
                    archive.unlink()
                exe = list(stage.rglob('llama-server.exe'))
                if exe:
                    # CUDA redistributable archive may contain runtime DLLs in
                    # a separate directory. Windows searches the EXE directory
                    # first, so stage required DLLs alongside llama-server.exe.
                    for dll in stage.rglob('*.dll'):
                        target = exe[0].parent / dll.name
                        if dll != target and not target.exists():
                            shutil.copy2(dll, target)
                if not exe:
                    raise RuntimeError(f'No llama-server.exe in official {tag} archives.')
                (stage / 'grand-council-runtime.json').write_text(
                    json.dumps({'tag': tag, 'backend': backend, 'assets': [a['name'] for a in assets]}, indent=2))
                # Atomically publish only verified, extracted assets.
                os.replace(stage, destination)
            progress(f'{backend} inference engine installed ({tag}).')
            return self.find_exe(backend)

    def shutdown(self):
        with self.lock:
            for process in self.processes:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        process.kill()
            self.processes.clear()


class LlamaServer:
    """Owns an isolated llama-server process with an ephemeral loopback port."""
    def __init__(self, manager, model_path, context, gpu_layers=-1, backend=None, progress=None):
        self.manager = manager
        self.process = None
        self.logfile = None
        self.backend = None
        self.url = None
        self.context = context
        preferred = [backend] if backend else hardware_profile()['preferred']
        errors = []
        for name in preferred:
            try:
                self._start(name, model_path, context, gpu_layers, progress)
                return
            except Exception as exc:
                errors.append(f'{name}: {exc}')
                self.close()
        raise RuntimeError('Could not start local GGUF inference. ' + ' | '.join(errors))

    def _start(self, backend, model_path, context, gpu_layers, progress):
        exe = self.manager.find_exe(backend)
        if exe is None:
            exe = self.manager.install(backend, progress=progress)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        args = [str(exe), '-m', str(model_path), '--host', '127.0.0.1', '--port', str(port),
                '--ctx-size', str(context), '--n-gpu-layers', str(gpu_layers if backend != 'cpu' else 0),
                '--parallel', '1']
        logdir = self.manager.home / 'logs'
        logdir.mkdir(parents=True, exist_ok=True)
        self.logfile = open(logdir / f'llama-server-{port}.log', 'ab')
        options = {'stdout': self.logfile, 'stderr': subprocess.STDOUT,
                   'cwd': str(exe.parent)}
        if os.name == 'nt':
            options['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        self.process = subprocess.Popen(args, **options)
        self.manager.processes.append(self.process)
        self.backend = backend
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f'llama-server exited {self.process.returncode}; check {self.logfile.name}')
            try:
                result = httpx.get(self.url + '/health', timeout=2)
                if result.status_code == 200:
                    return
            except (httpx.TransportError, ValueError):
                pass
            time.sleep(0.4)
        raise RuntimeError('llama-server did not become healthy within 90 seconds.')

    def generate(self, messages, max_tokens, temperature):
        response = httpx.post(self.url + '/v1/chat/completions', json={
            'model': 'local', 'messages': messages, 'max_tokens': max_tokens,
            'temperature': temperature}, timeout=600)
        response.raise_for_status()
        body = response.json()
        choice = body['choices'][0]
        return {'text': choice['message'].get('content') or '',
                'finish_reason': choice.get('finish_reason') or 'unknown',
                'output_tokens': (body.get('usage') or {}).get('completion_tokens'),
                'applied_max_tokens': max_tokens, 'accelerator': self.backend}

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.logfile and not self.logfile.closed:
            self.logfile.close()
