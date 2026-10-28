"""No network, GPU drivers, or Windows machine required."""
import io
import json
import os
import zipfile

import pytest

from grand_council import acceleration as a
from grand_council.models import ModelEngine


def asset(name):
    return {'name': name, 'size': 4096,
            'digest': 'sha256:' + '0'*64,
            'browser_download_url': 'https://github.com/ggml-org/llama.cpp/releases/download/b1/'+name}


def release(backend='cuda12'):
    names = {
        'cuda12': ['llama-b1-bin-win-cuda-12.4-x64.zip', 'cudart-llama-bin-win-cuda-12.4-x64.zip'],
        'vulkan': ['llama-b1-bin-win-vulkan-x64.zip'],
        'cpu': ['llama-b1-bin-win-cpu-x64.zip']}
    return {'tag_name': 'b1', 'assets': [asset(x) for x in names[backend]]}


def test_select_release_skips_nonbinary_latest():
    tag, found = a.select_release([{'tag_name':'v0.7.0', 'assets':[asset('nightly-tag.txt')]},
                                   release('cuda12')], 'cuda12')
    assert tag == 'b1'
    assert len(found) == 2


def test_requires_cuda_redistributables():
    data = release('cuda12')
    data['assets'] = data['assets'][:1]
    with pytest.raises(RuntimeError, match='compatible'):
        a.select_release([data], 'cuda12')


def test_reject_unverified_or_unofficial_binary(tmp_path):
    suspicious = asset('llama.zip')
    suspicious['digest'] = None
    with pytest.raises(RuntimeError, match='SHA256'):
        a._download_verified(suspicious, tmp_path/'a.zip')
    suspicious['digest'] = 'sha256:'+'0'*64
    suspicious['browser_download_url'] = 'https://example.org/evil.zip'
    with pytest.raises(RuntimeError, match='location'):
        a._download_verified(suspicious, tmp_path/'a.zip')


def test_archive_traversal_is_rejected(tmp_path):
    bad = tmp_path/'bad.zip'
    with zipfile.ZipFile(bad, 'w') as z:
        z.writestr('../outside.exe', 'evil')
    with pytest.raises(RuntimeError, match='Unsafe'):
        a._safe_extract(bad, tmp_path/'out')
    assert not (tmp_path/'outside.exe').exists()


def test_hardware_prefers_cuda12_for_gtx1080(monkeypatch):
    monkeypatch.setattr(a.platform, 'system', lambda:'Windows')
    monkeypatch.setattr(a.platform, 'machine', lambda:'AMD64')
    def fake(args, timeout=8):
        if args[0] == 'nvidia-smi':
            return 'GeForce GTX 1080, 8192, 560.70'
        return ''
    monkeypatch.setattr(a, '_command', fake)
    profile = a.hardware_profile()
    assert profile['preferred'] == ['cuda12', 'vulkan', 'cpu']
    assert profile['gpu'][0]['vram_gb'] == 8.0


def test_hardware_old_nvidia_driver_uses_vulkan(monkeypatch):
    monkeypatch.setattr(a.platform, 'system', lambda:'Windows')
    monkeypatch.setattr(a, '_command', lambda args, timeout=8:
                        'GeForce GTX 1080, 8192, 536.00' if args[0] == 'nvidia-smi' else '')
    assert a.hardware_profile()['preferred'] == ['vulkan', 'cpu']


def test_runtime_install_extracts_and_reuses_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(a.platform, 'system', lambda:'Windows')
    monkeypatch.setattr(a.platform, 'machine', lambda:'AMD64')
    def fake_download(asset, destination):
        with zipfile.ZipFile(destination,'w') as z:
            if asset['name'].startswith('cudart-'):
                z.writestr('lib/cudart64_12.dll', 'dll')
            else:
                z.writestr('bin/llama-server.exe', 'exe')
    monkeypatch.setattr(a, '_download_verified', fake_download)
    manager = a.RuntimeManager(tmp_path)
    exe = manager.install('cuda12', releases=[release('cuda12')])
    assert exe.is_file()
    assert (exe.parent/'cudart64_12.dll').exists()
    assert manager.install('cuda12', releases=[]) == exe


def test_unload_stops_server():
    class FakeServer:
        def __init__(self): self.closed = False
        def close(self): self.closed = True
    server = FakeServer()
    engine = ModelEngine()
    engine.cache['a'] = ('llama_server', server)
    engine.order.append('a')
    engine.unload()
    assert server.closed


def test_managed_engine_routes_to_server(tmp_path, monkeypatch):
    from grand_council import models
    model_file=tmp_path/'tiny.gguf';model_file.write_bytes(b'GGUF')
    class FakeLlama:
        def __init__(self, *args, **kwargs): pass
        def generate(self, messages, max_tokens, temperature):
            return {'text':'GPU response', 'finish_reason':'stop'}
        def close(self): pass
    monkeypatch.setattr(a, 'LlamaServer', FakeLlama)
    engine = ModelEngine(runtime_manager=object())
    cfg = {'id':'test','backend':'gguf','local_path':str(model_file),'context':2048}
    assert engine.generate(cfg,[{'role':'user','content':'test'}])['text'] == 'GPU response'
