import time
from pathlib import Path

from fastapi.testclient import TestClient

from grand_council.council import parse_report, to_markdown
from grand_council.models import check_repo
from grand_council.server import create_app


def test_repo_validation():
    assert check_repo('Qwen/Qwen3-0.6B') == 'Qwen/Qwen3-0.6B'
    for repo in ('', '../bad', 'a/../../secret', 'foo bar', '/root', 'abc\\def'):
        try:
            check_repo(repo)
            assert False, repo
        except ValueError:
            pass


def test_report_parser_handles_code_fences():
    assert parse_report('```json\n{"findings":[],"executive_summary":"OK"}\n```')['executive_summary'] == 'OK'
    assert parse_report('unstructured output')['parse_warning']


def test_full_demo_council(tmp_path):
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        assert 'Grand Council' in client.get('/').text
        assert client.get('/api/health').json()['ok']
        assert client.get('/api/models').json()[0]['id'] == 'demo'
        assert client.get('/api/hardware').status_code == 200
        agents = [
            {'id':'architect','name':'Architect','role':'Design','model_id':'demo','max_tokens':500},
            {'id':'skeptic','name':'Skeptic','role':'Challenge','model_id':'demo','max_tokens':500},
        ]
        payload = {'title':'Test spec','spec':'Build a safe and reliable offline todo app.',
                   'agents':agents,'chair_id':'architect','rounds':1,
                   'criteria':['Security','Usability'],'input_budget':6000}
        res = client.post('/api/councils',json=payload)
        assert res.status_code == 200, res.text
        ident = res.json()['id']
        for _ in range(120):
            item = client.get('/api/councils/'+ident).json()
            if item['status'] not in ('queued','running'):
                break
            time.sleep(.05)
        assert item['status'] == 'completed', item.get('error')
        assert len(item['messages']) == 7
        assert all('[SIMULATED DEMO' in m['text'] or '"findings"' in m['text'] for m in item['messages'])
        assert item['report']['findings'][0]['severity'] == 'high'
        assert item['warnings']
        assert client.get('/api/councils').json()[0]['id'] == ident
        md=client.get('/api/councils/'+ident+'/export?format=md')
        assert md.status_code == 200
        assert 'Original specification' in md.text
        assert 'safe and reliable' in md.text
        data=client.get('/api/councils/'+ident+'/export?format=json')
        assert data.status_code==200
        assert '"messages"' in data.text
    # Persistence across application restarts.
    second = create_app(data_dir=tmp_path)
    with TestClient(second) as client:
        assert client.get('/api/councils/'+ident).json()['status']=='completed'


def test_invalid_agents_and_endpoint(tmp_path):
    app=create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        bad=client.post('/api/councils',json={'spec':'abc','agents':[]})
        assert bad.status_code==422
        invalid=client.post('/api/models/install',json={'repo':'../../etc/passwd','backend':'gguf','filename':'foo.gguf'})
        assert invalid.status_code==422
        endpoint=client.post('/api/models/endpoint',json={'base_url':'http://127.0.0.1:1234/v1','model_name':'qwen'})
        assert endpoint.status_code==200
        identifier=endpoint.json()['id']
        assert client.post('/api/models/'+identifier+'/load').status_code==200
        assert client.delete('/api/models/'+identifier).status_code==200
        assert all(m['id']!=identifier for m in client.get('/api/models').json())

class SpyEngine:
    def __init__(self):
        self.calls=[]
        self.max_loaded=1
    def loaded(self):return []
    def ensure(self, cfg):return None
    def unload(self, model_id=None):return None
    def generate(self, cfg, messages, max_tokens=384, temperature=.5):
        prompt=messages[-1]['content']
        self.calls.append((cfg['id'],prompt))
        if 'RETURN VALID JSON ONLY' in prompt:
            return '{"executive_summary":"Two independent endpoint assignments were exercised", "findings":[],"agreement":[],"dissent":[],"open_questions":[],"next_steps":[]}'
        return 'Detailed technical critique by '+cfg['id']+'. This proposal questions tradeoffs.'


def test_multi_model_assignment_and_two_debate_rounds(tmp_path):
    spy=SpyEngine()
    app=create_app(data_dir=tmp_path,engine=spy)
    with TestClient(app) as client:
        m1=client.post('/api/models/endpoint',json={'base_url':'http://127.0.0.1:1111/v1','model_name':'first'}).json()['id']
        m2=client.post('/api/models/endpoint',json={'base_url':'http://127.0.0.1:2222/v1','model_name':'second'}).json()['id']
        agents=[{'id':'one','name':'One','role':'Architecture','model_id':m1},
                {'id':'two','name':'Two','role':'Testing','model_id':m2}]
        response=client.post('/api/councils',json={'title':'multi-backend','spec':'The full specification goes here.',
                             'agents':agents,'chair_id':'one','rounds':2})
        assert response.status_code==200,response.text
        ident=response.json()['id']
        for _ in range(100):
            run=client.get('/api/councils/'+ident).json()
            if run['status'] in ('completed','failed'):
                break
            time.sleep(.03)
        assert run['status']=='completed',run.get('error')
        assert len(spy.calls)==11
        assert {x[0] for x in spy.calls}=={m1,m2}
        assert 'PEER POSITIONS' in spy.calls[2][1]
        assert 'CRITIQUES' in spy.calls[4][1]
        assert 'FINAL POSITIONS' in spy.calls[-1][1]
        assert 'full specification' in spy.calls[-1][1].lower()


def test_runtime_install_plans_and_disk_message():
    from grand_council.runtime_installer import (
        install_commands, pip_failure_message, require_disk_room,
        storage_report, CPU_WHEEL_INDEX,
    )
    import pytest
    from tempfile import TemporaryDirectory

    cpu = install_commands('transformers-cpu', 'python.exe')
    assert len(cpu) == 2
    assert CPU_WHEEL_INDEX in cpu[0]
    assert cpu[0][-1] == 'torch>=2.2'
    assert all('--no-cache-dir' in cmd for cmd in cpu)
    assert 'transformers>=4.48,<6' in cpu[1]
    assert CPU_WHEEL_INDEX not in cpu[1]
    assert 'torch>=2.2' not in cpu[1]
    assert len(install_commands('gguf', 'python.exe')) == 1
    assert len(install_commands('transformers', 'python.exe')) == 1
    with pytest.raises(ValueError):
        install_commands('not-valid', 'python.exe')
    report = {'python_environment': {'free_gb': 0.3},
              'temporary_files': {'free_gb': 20.0}}
    with pytest.raises(RuntimeError, match='Not enough disk space'):
        require_disk_room('transformers-cpu', report)
    assert 'ran out of disk space' in pip_failure_message(
        1, ['ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device'])
    assert 'pip exited with code 1' in pip_failure_message(1, ['Other issue'])
    with TemporaryDirectory() as directory:
        info = storage_report(directory, directory, directory)
        assert info['python_environment']['free_gb'] is not None
        assert info['temporary_files']['free_gb'] is not None
        assert info['model_library']['free_gb'] is not None


def test_runtime_install_rejects_unknown_modes(tmp_path):
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        assert client.post('/api/runtimes/install', json={'backend':'arbitrary-pip-package'}).status_code == 422
        assert client.get('/api/hardware').json()['storage']['python_environment']['free_gb'] is not None
