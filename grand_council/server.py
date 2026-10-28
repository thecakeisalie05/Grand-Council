"""Local-only API and static web frontend for Grand Council."""
import os
import sys
import platform
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .acceleration import RuntimeManager, hardware_profile
from .council import CouncilRunner, now, to_markdown, RESPONSE_BUDGETS
from .models import ModelEngine, check_repo, inspect_hub
from .runtime_installer import install_commands, storage_report, require_disk_room, pip_failure_message
from .store import Store

STATIC = Path(__file__).resolve().parent / 'static'
DEMO = {'id': 'demo', 'label': 'Simulation · no model', 'backend': 'demo',
        'repo': None, 'status': 'ready', 'context': 8192}


def create_app(data_dir=None, engine=None):
    home = Path(data_dir or os.environ.get('GRAND_COUNCIL_HOME', '~/.grand-council')).expanduser().resolve()
    store = Store(home)
    runtime_manager = RuntimeManager(home)
    engine = engine or ModelEngine(runtime_manager=runtime_manager if platform.system() == 'Windows' else None)
    models = {m['id']: m for m in store.get('models', [])}
    models['demo'] = dict(DEMO)
    jobs = {}
    jobs_lock = threading.RLock()
    model_lock = threading.RLock()
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='council')
    runner = CouncilRunner(store, models, engine)
    # A process restart cannot resume in-flight model generations.
    for item in store.list_runs():
        if item['status'] in ('running', 'queued'):
            previous = store.get_run(item['id'])
            if previous:
                previous['status'] = 'interrupted'
                previous['phase'] = 'Interrupted by server restart'
                previous['error'] = 'The server restarted before this council completed.'
                previous['updated_at'] = now()
                store.put_run(previous)
    app = FastAPI(title='Grand Council', version='0.2.0', docs_url='/api/docs')
    app.state.runner = runner
    app.state.models = models
    app.state.jobs = jobs
    app.state.engine = engine
    app.state.pool = pool
    app.state.runtime_manager = runtime_manager

    def persist_models():
        with model_lock:
            store.put('models', [v for k, v in models.items() if k != 'demo'])

    def job_update(job_id, **kwargs):
        with jobs_lock:
            jobs[job_id].update(kwargs)
            jobs[job_id]['updated_at'] = now()

    def start_job(kind, fn):
        jid = uuid4().hex[:12]
        with jobs_lock:
            jobs[jid] = {'id': jid, 'type': kind, 'status': 'queued', 'message': 'Queued',
                         'created_at': now(), 'updated_at': now()}
        def wrapped():
            try:
                job_update(jid, status='running')
                result = fn(jid)
                job_update(jid, status='completed', message='Complete', result=result)
            except Exception as exc:
                job_update(jid, status='failed', message=f'{type(exc).__name__}: {exc}')
        pool.submit(wrapped)
        return {'job_id': jid}

    @app.get('/api/health')
    def health():
        return {'application': 'grand-council', 'ok': True, 'version': '0.2.0', 'active_run': runner.active,
                'max_loaded': engine.max_loaded, 'loaded': engine.loaded()}

    @app.get('/api/hardware')
    def hardware():
        accelerator = hardware_profile()
        info = {'platform': platform.platform(), 'cpu_cores': psutil.cpu_count(logical=True),
                'ram_total_gb': round(psutil.virtual_memory().total / 1024**3, 1),
                'ram_available_gb': round(psutil.virtual_memory().available / 1024**3, 1),
                'gpu': accelerator['gpu'], 'accelerator': accelerator,
                'managed_runtimes': runtime_manager.installed(), 'desktop': getattr(sys, 'frozen', False),
                'runtimes': {},
                'storage': storage_report(os.sys.prefix, model_dir=home / 'models')}
        try:
            import torch
            info['runtimes']['torch'] = torch.__version__
            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(idx)
                    if not any(g['name'] == p.name for g in info['gpu']):
                        info['gpu'].append({'name': p.name, 'vram_gb': round(p.total_memory / 1024**3, 1)})
        except Exception:
            info['runtimes']['torch'] = None
        try:
            import transformers
            info['runtimes']['transformers'] = transformers.__version__
        except Exception:
            info['runtimes']['transformers'] = None
        try:
            import llama_cpp
            info['runtimes']['llama_cpp'] = getattr(llama_cpp, '__version__', 'installed')
        except Exception:
            info['runtimes']['llama_cpp'] = None
        return info

    @app.get('/api/models')
    def get_models():
        with model_lock:
            return [{**m, 'loaded': mid in engine.loaded()} for mid, m in models.items()]

    @app.post('/api/models/inspect')
    def inspect(body: dict):
        try:
            return inspect_hub(body.get('repo', ''), body.get('token') or None,
                               body.get('revision') or None)
        except Exception as exc:
            raise HTTPException(400, f'Hugging Face lookup failed: {exc}')

    @app.get('/api/jobs')
    def get_jobs():
        with jobs_lock:
            return sorted([dict(x) for x in jobs.values()], key=lambda x: x['created_at'], reverse=True)

    @app.get('/api/jobs/{job_id}')
    def get_job(job_id: str):
        with jobs_lock:
            if job_id not in jobs:
                raise HTTPException(404, 'Job not found')
            return dict(jobs[job_id])

    @app.post('/api/models/install')
    def install(body: dict):
        try:
            repo = check_repo(body.get('repo', ''))
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        backend = body.get('backend')
        if backend not in ('gguf', 'transformers'):
            raise HTTPException(422, 'Choose GGUF or Transformers backend.')
        filename = body.get('filename', '')
        if backend == 'gguf' and (not filename.lower().endswith('.gguf') or '..' in filename):
            raise HTTPException(422, 'Select a valid GGUF filename from the inspected repository.')
        if bool(body.get('trust_remote_code')) and backend != 'transformers':
            raise HTTPException(422, 'Trust remote code only applies to Transformers.')
        token = body.get('token') or os.environ.get('HF_TOKEN')
        revision = body.get('revision') or None
        context = max(512, min(131072, int(body.get('context', 8192))))
        gpu_layers = max(-1, min(1000, int(body.get('gpu_layers', -1))))
        mid = 'hf_' + uuid4().hex[:12]
        folder = home / 'models' / mid
        label = str(body.get('label') or repo.rsplit('/', 1)[-1])[:120]

        def do_install(jid):
            job_update(jid, message='Inspecting repository and checking file formats…')
            info = HfApi(token=token).model_info(repo, revision=revision)
            names = [x.rfilename for x in (info.siblings or [])]
            commit = info.sha
            if backend == 'gguf':
                if filename not in names:
                    raise ValueError(f'File not found in repository: {filename}')
                shards = [filename]
                match = re.search(r'-\d{5}-of-\d{5}\.gguf$', filename)
                if match:
                    prefix = filename[:match.start()]
                    suffix = filename[match.start():]
                    total = suffix.split('-of-')[1].split('.')[0]
                    shards = sorted(n for n in names if n.startswith(prefix + '-')
                                    and n.endswith(f'-of-{total}.gguf'))
                job_update(jid, message=f'Downloading {len(shards)} GGUF file(s)…')
                folder.mkdir(parents=True, exist_ok=True)
                for i, shard in enumerate(shards, 1):
                    job_update(jid, message=f'Downloading GGUF file {i}/{len(shards)}: {shard}')
                    hf_hub_download(repo_id=repo, filename=shard, revision=commit,
                                    token=token, local_dir=str(folder))
                path = str(folder / shards[0])
                if not Path(path).is_file():
                    raise RuntimeError('Download incomplete: GGUF file missing.')
            else:
                safe = [n for n in names if n.endswith('.safetensors')]
                if not safe:
                    raise ValueError('No safetensors weights found. Convert the model to safetensors '
                                     'or use a GGUF repository. Legacy pickle weights are deliberately blocked.')
                patterns = ['*.safetensors', '*.json', '*.model', '*.tiktoken', '*.txt',
                            '*.vocab', '*.merges', '*.jinja', '*.sentencepiece']
                if body.get('trust_remote_code'):
                    patterns += ['*.py']
                job_update(jid, message=f'Downloading Transformers checkpoint ({len(safe)} weight files)…')
                path = snapshot_download(repo_id=repo, revision=commit, token=token,
                                         local_dir=str(folder), allow_patterns=patterns)
            record = {'id': mid, 'label': label, 'repo': repo, 'backend': backend,
                      'filename': filename if backend == 'gguf' else None,
                      'revision': commit, 'local_path': str(path), 'status': 'ready',
                      'context': context, 'gpu_layers': gpu_layers,
                      'trust_remote_code': bool(body.get('trust_remote_code', False)),
                      'installed_at': now()}
            with model_lock:
                models[mid] = record
                persist_models()
            return {'model_id': mid, 'label': label}
        return start_job('download', do_install)

    @app.post('/api/models/endpoint')
    def add_endpoint(body: dict):
        base_url = str(body.get('base_url') or '').strip().rstrip('/')
        model_name = str(body.get('model_name') or '').strip()
        if not (base_url.startswith('http://') or base_url.startswith('https://')) or not model_name:
            raise HTTPException(422, 'Provide a full HTTP(S) base URL and a model name.')
        mid = 'endpoint_' + uuid4().hex[:12]
        cfg = {'id': mid, 'label': str(body.get('label') or model_name)[:120],
               'backend': 'openai', 'base_url': base_url, 'model_name': model_name,
               'api_key_env': 'COUNCIL_API_KEY', 'context': max(512, min(131072, int(body.get('context', 8192)))),
               'status': 'ready', 'installed_at': now()}
        with model_lock:
            models[mid] = cfg
            persist_models()
        return cfg

    @app.post('/api/runtimes/auto')
    def install_automatic_runtime():
        profile = hardware_profile()
        if platform.system() != 'Windows':
            raise HTTPException(422, 'Automatic runtime setup currently targets Windows x64.')
        def task(jid):
            errors = []
            for backend in profile['preferred']:
                try:
                    exe = runtime_manager.install(backend, progress=lambda msg: job_update(jid, message=msg))
                    return {'backend': backend, 'executable': str(exe)}
                except Exception as exc:
                    errors.append(f'{backend}: {exc}')
                    job_update(jid, message=f'{backend} unavailable; trying fallback…')
            raise RuntimeError('All official llama.cpp backends failed: ' + ' | '.join(errors))
        return start_job('auto-runtime-install', task)

    @app.post('/api/runtimes/install')
    def install_runtime(body: dict):
        """Explicit, allowlisted package installation; no arbitrary shell commands."""
        import subprocess
        import sys
        if getattr(sys, 'frozen', False):
            raise HTTPException(422, 'Packaged desktop builds use automatic llama.cpp runtime setup, not pip. '                                       'Use the Auto GPU setup button for GGUF models.')
        kind = body.get('backend')
        if kind not in ('gguf', 'transformers-cpu', 'transformers'):
            raise HTTPException(422, 'Choose gguf, transformers-cpu, or transformers.')

        def task(jid):
            report = storage_report(sys.prefix)
            require_disk_room(kind, report)
            commands = install_commands(kind, sys.executable)
            for idx, command in enumerate(commands, 1):
                job_update(jid, message=f'Installing {kind} (step {idx}/{len(commands)}) with pip cache disabled…')
                process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True,
                                           encoding='utf-8', errors='replace', bufsize=1)
                tail = []
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        tail.append(line)
                        tail = tail[-8:]
                        job_update(jid, message=line[:240])
                code = process.wait()
                if code != 0:
                    raise RuntimeError(pip_failure_message(code, tail))
            return {'runtime': kind, 'restart_required': True}
        return start_job('runtime-install', task)

    @app.post('/api/models/{model_id}/load')
    def preload(model_id: str):
        with model_lock:
            if model_id not in models:
                raise HTTPException(404, 'Model not found')
            cfg = dict(models[model_id])
        def task(jid):
            job_update(jid, message=f'Loading {cfg["label"]} into memory…')
            engine.ensure(cfg)
            return {'model_id': model_id}
        return start_job('load', task)

    @app.post('/api/models/unload')
    def unload_models():
        engine.unload()
        return {'loaded': []}

    @app.put('/api/models/cache')
    def set_cache(body: dict):
        count = int(body.get('max_loaded', 1))
        if count < 1 or count > 5:
            raise HTTPException(422, 'Resident models must be between 1 and 5.')
        engine.max_loaded = count
        while len(engine.loaded()) > count:
            engine.unload(engine.loaded()[0])
        return {'max_loaded': count}

    @app.delete('/api/models/{model_id}')
    def remove_model(model_id: str):
        if model_id == 'demo':
            raise HTTPException(422, 'The demo adapter is built in.')
        with model_lock:
            cfg = models.get(model_id)
            if not cfg:
                raise HTTPException(404, 'Model not found')
            if runner.active:
                raise HTTPException(409, 'Cannot remove models during an active council.')
            engine.unload(model_id)
            models.pop(model_id, None)
            persist_models()
        path = home / 'models' / model_id
        if path.exists() and path.is_dir() and path.parent == (home / 'models'):
            shutil.rmtree(path)
        return {'removed': model_id}

    @app.get('/api/councils')
    def list_councils():
        return store.list_runs()

    @app.post('/api/councils')
    def create_council(body: dict):
        spec = str(body.get('spec') or '').strip()
        if not spec or len(spec) > 500000:
            raise HTTPException(422, 'Specification required (1–500,000 characters).')
        agents = body.get('agents', [])
        if not isinstance(agents, list) or not 2 <= len(agents) <= 12:
            raise HTTPException(422, 'Add 2–12 council agents.')
        seen = set()
        valid_agents = []
        for i, a in enumerate(agents):
            if not isinstance(a, dict):
                raise HTTPException(422, 'Invalid agent definition.')
            mid = a.get('model_id')
            if mid not in models:
                raise HTTPException(422, f'Agent {i+1}: select an installed model or endpoint.')
            aid = str(a.get('id') or '')
            if not aid or aid in seen:
                raise HTTPException(422, 'Each agent requires a unique ID.')
            seen.add(aid)
            valid_agents.append({'id': aid, 'name': str(a.get('name') or f'Agent {i+1}')[:80],
                                 'role': str(a.get('role') or 'Independent evaluator')[:1200],
                                 'model_id': mid,
                                 'temperature': max(0, min(1.5, float(a.get('temperature', .45)))),
                                 **({'max_tokens': max(128, min(4096, int(a['max_tokens'])))} if 'max_tokens' in a else {})})
        chair_id = body.get('chair_id')
        if chair_id not in seen:
            raise HTTPException(422, 'Chair must be one of the active agents.')
        rounds = int(body.get('rounds', 1))
        if not 0 <= rounds <= 5:
            raise HTTPException(422, 'Debate rounds must be between 0 and 5.')
        input_budget = max(4000, min(60000, int(body.get('input_budget', 10000))))
        detail = str(body.get('response_detail') or 'balanced').lower()
        if detail not in RESPONSE_BUDGETS:
            raise HTTPException(422, 'Response detail must be concise, balanced, or thorough.')
        auto_continue = body.get('auto_continue', True)
        if not isinstance(auto_continue, bool):
            raise HTTPException(422, 'Auto-continue must be true or false.')
        try:
            max_continuations = int(body.get('max_continuations', 1))
        except (ValueError, TypeError):
            raise HTTPException(422, 'Continuation limit must be 0, 1, or 2.')
        if max_continuations not in (0, 1, 2):
            raise HTTPException(422, 'Continuation limit must be 0, 1, or 2.')
        criteria = body.get('criteria') or []
        if not isinstance(criteria, list):
            raise HTTPException(422, 'Review criteria must be an array.')
        run = {'id': uuid4().hex[:14], 'title': str(body.get('title') or 'Untitled Council')[:160],
               'spec': spec, 'agents': valid_agents, 'chair_id': chair_id,
               'rounds': rounds, 'input_budget': input_budget,
               'response_detail': detail, 'auto_continue': auto_continue,
               'max_continuations': max_continuations,
               'criteria': [str(x)[:300] for x in criteria[:15]],
               'created_at': now(), 'updated_at': now(), 'phase': 'Queued',
               'status': 'queued', 'messages': [], 'warnings': [], 'report': None, 'error': None}
        if any(a['model_id'] == 'demo' for a in valid_agents):
            run['warnings'].append('Demo adapter used: its messages are simulated placeholders, not AI evaluations.')
        try:
            runner.launch(run, pool)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {'id': run['id'], 'status': run['status']}

    @app.get('/api/councils/{run_id}')
    def get_council(run_id: str):
        if not re.fullmatch('[a-f0-9]{14}', run_id):
            raise HTTPException(404, 'Council not found')
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(404, 'Council not found')
        return run

    @app.post('/api/councils/{run_id}/cancel')
    def cancel_council(run_id: str):
        if not runner.cancel(run_id):
            raise HTTPException(409, 'This council is not active.')
        return {'cancel_requested': True, 'note': 'Takes effect after the current model response.'}

    @app.get('/api/councils/{run_id}/export')
    def export_council(run_id: str, format: str = 'md'):
        run = store.get_run(run_id) if re.fullmatch('[a-f0-9]{14}', run_id) else None
        if not run:
            raise HTTPException(404, 'Council not found')
        if format == 'md':
            return PlainTextResponse(to_markdown(run), headers={'Content-Disposition':
                                     f'attachment; filename="council-{run_id}.md"'},
                                     media_type='text/markdown; charset=utf-8')
        if format == 'json':
            import json
            return PlainTextResponse(json.dumps(run, ensure_ascii=False, indent=2),
                                     headers={'Content-Disposition': f'attachment; filename="council-{run_id}.json"'},
                                     media_type='application/json')
        raise HTTPException(422, 'Use md or json format.')

    @app.get('/')
    def root():
        return FileResponse(STATIC / 'index.html')

    app.mount('/static', StaticFiles(directory=str(STATIC)), name='static')
    return app


app = create_app()


def main():
    import uvicorn
    uvicorn.run('grand_council.server:app', host='127.0.0.1', port=int(os.environ.get('GRAND_COUNCIL_PORT', '8765')),
                reload=False)


if __name__ == '__main__':
    main()
