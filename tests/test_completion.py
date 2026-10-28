"""End-to-end completion accounting, continuation policy, and budget controls."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from grand_council.council import CouncilRunner, merge_continuation, RESPONSE_BUDGETS
from grand_council.models import ModelEngine
from grand_council.server import create_app
from grand_council.store import Store


class QueueEngine:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, cfg, messages, max_tokens, temperature):
        self.calls.append({'cfg': cfg, 'messages': messages, 'budget': max_tokens})
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def runner_with_engine(tmp_path, engine):
    store = Store(tmp_path)
    model = {'id': 'm', 'label': 'Fake adapter', 'backend': 'test', 'context': 8192}
    runner = CouncilRunner(store, {'m': model}, engine)
    agent = {'id': 'one', 'name': 'Architect', 'role': 'Architecture', 'model_id': 'm', 'temperature': 0.4}
    run = {'id': 'fake123', 'messages': [], 'warnings': [], 'response_detail': 'balanced',
           'auto_continue': True, 'max_continuations': 1}
    return runner, run, agent


def test_phase_specific_budgets_and_length_continuation(tmp_path):
    engine = QueueEngine([
        {'text': 'The first par', 'finish_reason': 'length', 'output_tokens': 1536},
        {'text': 't concludes normally.', 'finish_reason': 'stop', 'output_tokens': 13},
    ])
    runner, run, agent = runner_with_engine(tmp_path, engine)
    message = runner._say(run, 'Independent proposals', agent, 'Review architecture')
    assert message['text'] == 'The first part concludes normally.'
    assert message['completion_status'] == 'complete'
    assert message['generation_calls'] == 2
    assert message['output_tokens'] == 1549
    assert message['per_call_budget'] == RESPONSE_BUDGETS['balanced']['proposal']
    assert engine.calls[0]['messages'][-1]['content'] == 'Review architecture'
    assert engine.calls[1]['messages'][-2] == {'role':'assistant','content':'The first par'}
    assert 'Continue exactly' in engine.calls[1]['messages'][-1]['content']
    assert not run['warnings']


def test_unfinished_output_is_marked_and_persisted(tmp_path):
    engine = QueueEngine([
        {'text': 'Introduction. ', 'finish_reason': 'length', 'output_tokens': 896},
        {'text': 'More findings but not the end', 'finish_reason': 'length', 'output_tokens': 896},
    ])
    runner, run, agent = runner_with_engine(tmp_path, engine)
    msg = runner._say(run, 'Round 1 · critiques', agent, 'Peer positions')
    assert msg['completion_status'] == 'partial'
    assert msg['generation_calls'] == 2
    assert msg['per_call_budget'] == RESPONSE_BUDGETS['balanced']['critique']
    assert len(run['warnings']) == 1
    assert 'Output still reached' in msg['completion_warning']
    assert runner.store.get_run(run['id'])['messages'][0]['completion_status'] == 'partial'


def test_context_failure_during_continuation_keeps_first_chunk(tmp_path):
    engine = QueueEngine([
        {'text': 'First portion.', 'finish_reason': 'length', 'output_tokens': 1200},
        RuntimeError('Context exhausted'),
    ])
    runner, run, agent = runner_with_engine(tmp_path, engine)
    msg = runner._say(run, 'Independent proposals', agent, 'Review')
    assert msg['text'] == 'First portion.'
    assert msg['generation_calls'] == 1
    assert msg['completion_status'] == 'partial'
    assert 'Context exhausted' in msg['completion_warning']


def test_initial_backend_error_is_not_hidden(tmp_path):
    engine = QueueEngine([RuntimeError('Provider unreachable')])
    runner, run, agent = runner_with_engine(tmp_path, engine)
    with pytest.raises(RuntimeError, match='Provider unreachable'):
        runner._say(run, 'Independent proposals', agent, 'Review')


def test_old_backend_unknown_finish_does_not_trigger_fake_continuation(tmp_path):
    engine = QueueEngine(['Legacy backend result'])
    runner, run, agent = runner_with_engine(tmp_path, engine)
    msg = runner._say(run, 'Independent proposals', agent, 'Review')
    assert msg['completion_status'] == 'unknown'
    assert msg['generation_calls'] == 1
    assert not run['warnings']


def test_overlap_and_disabled_automatic_continuation(tmp_path):
    assert merge_continuation('The outcome is ready', 'outcome is ready now') == 'The outcome is ready now'
    engine = QueueEngine([{'text': 'Cut off.', 'finish_reason': 'length', 'output_tokens': 20}])
    runner, run, agent = runner_with_engine(tmp_path, engine)
    run['auto_continue'] = False
    msg = runner._say(run, 'Independent proposals', agent, 'Review')
    assert msg['completion_status'] == 'partial'
    assert msg['generation_calls'] == 1


def test_api_validates_response_detail_and_preserves_old_agent_definitions(tmp_path):
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        base = {'title': 'Quality', 'spec':'Review a specification', 'agents':[
            {'id':'a','name':'A','model_id':'demo'}, {'id':'b','name':'B','model_id':'demo'}
        ], 'chair_id':'a', 'rounds':0}
        bad = client.post('/api/councils', json={**base,'response_detail':'infinite'})
        assert bad.status_code == 422
        bad = client.post('/api/councils', json={**base,'max_continuations':9})
        assert bad.status_code == 422
        bad = client.post('/api/councils', json={**base,'auto_continue':'yes'})
        assert bad.status_code == 422
        good = client.post('/api/councils', json={**base, 'response_detail':'thorough',
                                               'auto_continue':False, 'max_continuations':0})
        assert good.status_code == 200, good.text
        data = client.get('/api/councils/'+good.json()['id']).json()
        assert data['response_detail'] == 'thorough'
        assert data['auto_continue'] is False
        assert 'max_tokens' not in data['agents'][0]
        homepage = client.get('/').text
        js = client.get('/static/app.js').text
        assert 'RESPONSE DETAIL' in homepage
        assert 'INCOMPLETE RESPONSES' in homepage
        assert 'MAX TOKENS' not in js


def test_gguf_backend_context_adjusts_budget():
    class FakeLlama:
        def n_ctx(self): return 4096
        def tokenize(self, text, add_bos): return [1] * 2500
        def create_chat_completion(self, **kwargs):
            self.call = kwargs
            return {'choices':[{'message':{'content':'partial'},'finish_reason':'length'}],
                    'usage':{'completion_tokens':kwargs['max_tokens']}}
    fake = FakeLlama()
    engine=ModelEngine()
    engine.ensure=lambda cfg: ('gguf', fake)
    res=engine.generate({'id':'test','backend':'gguf','context':4096},
                        [{'role':'user','content':'Some spec'}],max_tokens=3000)
    assert res['applied_max_tokens'] == 4096-2500-128
    assert fake.call['max_tokens'] == res['applied_max_tokens']
    assert res['finish_reason'] == 'length'
