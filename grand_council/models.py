"""Inference backends, Hub inspection, and guarded local model loading."""
import gc
import os
import re
import threading
from pathlib import Path

REPO_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)?$')


def check_repo(repo):
    if not isinstance(repo, str) or not REPO_RE.fullmatch(repo) or '..' in repo or repo.startswith('.'):
        raise ValueError('Provide a valid Hugging Face model repository ID (owner/model).')
    return repo


def inspect_hub(repo, token=None, revision=None):
    from huggingface_hub import HfApi
    check_repo(repo)
    info = HfApi(token=token or os.environ.get('HF_TOKEN')).model_info(repo, revision=revision or None)
    files = [x.rfilename for x in (info.siblings or [])]
    ggufs = sorted(x for x in files if x.lower().endswith('.gguf'))
    safes = [x for x in files if x.lower().endswith('.safetensors')]
    config = info.config or {}
    architectures = config.get('architectures') or []
    pipeline = info.pipeline_tag or ''
    return {
        'repo': repo, 'revision': info.sha,
        'private': bool(info.private), 'gated': bool(info.gated),
        'architectures': architectures, 'pipeline': pipeline,
        'gguf_files': ggufs, 'safetensors': len(safes),
        'size_bytes': sum((getattr(x, 'size', 0) or 0) for x in (info.siblings or [])),
        'files': len(files),
        'backends': (['gguf'] if ggufs else []) + (['transformers'] if (config or safes) else []),
        'warning': ('Not all Hugging Face model architectures are text-generation models. '
                    'Transformers adapter requires a supported causal language model.')
    }


class ModelEngine:
    """Serializes inference; keeps at most max_loaded models in an LRU cache."""
    def __init__(self, max_loaded=1, runtime_manager=None):
        self.runtime_manager = runtime_manager
        self.lock = threading.RLock()
        self.cache = {}
        self.order = []
        self.max_loaded = max(1, int(max_loaded))

    def loaded(self):
        with self.lock:
            return list(self.order)

    def unload(self, model_id=None):
        with self.lock:
            ids = [model_id] if model_id else list(self.cache)
            for mid in ids:
                pair = self.cache.pop(mid, None)
                if mid in self.order:
                    self.order.remove(mid)
                if pair:
                    if pair[0] == 'llama_server':
                        pair[1].close()
                    del pair
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _load(self, config):
        kind = config['backend']
        if kind == 'demo':
            return ('demo', None)
        if kind == 'openai':
            return ('openai', None)
        path = Path(config.get('local_path', ''))
        if not path.exists():
            raise RuntimeError('Model not installed. Download it in Model Library first.')
        if kind == 'gguf':
            if self.runtime_manager is not None:
                from .acceleration import LlamaServer
                return ('llama_server', LlamaServer(
                    self.runtime_manager, str(path), int(config.get('context', 4096)),
                    int(config.get('gpu_layers', -1))))
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError('GGUF runtime missing. Install: pip install -e ".[gguf]"') from exc
            n_ctx = config.get('context', 4096)
            return ('gguf', Llama(model_path=str(path), n_ctx=n_ctx,
                                  n_gpu_layers=int(config.get('gpu_layers', -1)),
                                  verbose=False))
        if kind == 'transformers':
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError('Transformers runtime missing. Install: pip install -e ".[transformers]"') from exc
            trust = bool(config.get('trust_remote_code', False))
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=trust)
            kwargs = {'local_files_only': True, 'trust_remote_code': trust,
                      'device_map': 'auto', 'dtype': 'auto', 'use_safetensors': True,
                      'offload_folder': str(path / '.offload')}
            try:
                model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
            except TypeError as e:
                if 'dtype' in str(e):
                    kwargs['torch_dtype'] = kwargs.pop('dtype')
                    model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
                else:
                    raise
            model.eval()
            return ('transformers', (tokenizer, model, torch))
        raise ValueError(f'Unknown backend: {kind}')

    def ensure(self, cfg):
        mid = cfg['id']
        with self.lock:
            if mid not in self.cache:
                while len(self.cache) >= self.max_loaded:
                    self.unload(self.order[0])
                self.cache[mid] = self._load(cfg)
            if mid in self.order:
                self.order.remove(mid)
            self.order.append(mid)
            return self.cache[mid]

    def generate(self, cfg, messages, max_tokens=384, temperature=0.5):
        with self.lock:
            kind, engine = self.ensure(cfg)
            if kind == 'demo':
                return {'text': _demo_response(messages), 'finish_reason': 'stop', 'output_tokens': None}
            if kind == 'llama_server':
                return engine.generate(messages, max_tokens, temperature)
            if kind == 'gguf':
                # Reserve room for the prompt instead of attempting an impossible
                # per-call output length. llama.cpp owns final chat-template accounting.
                limit = int(cfg.get('context', engine.n_ctx()))
                serialized = '\n'.join(f"{msg['role']}: {msg['content']}" for msg in messages)
                try:
                    estimate = len(engine.tokenize(serialized.encode('utf-8'), add_bos=True))
                except (TypeError, AttributeError, ValueError):
                    estimate = (len(serialized.encode('utf-8')) + 2) // 3
                allocated = min(max_tokens, limit - estimate - 128)
                if allocated < 64:
                    raise RuntimeError(f'Prompt approximately uses {estimate} tokens; context of {limit} '
                                       'leaves insufficient room for a response. Reduce the spec/peer '
                                       'excerpt or increase this model\'s configured context.')
                out = engine.create_chat_completion(
                    messages=messages, max_tokens=allocated, temperature=temperature)
                choice = out['choices'][0]
                usage = out.get('usage') or {}
                return {'text': choice['message'].get('content') or '',
                        'finish_reason': choice.get('finish_reason') or 'unknown',
                        'output_tokens': usage.get('completion_tokens'), 'applied_max_tokens': allocated}
            if kind == 'transformers':
                tokenizer, model, torch = engine
                if tokenizer.chat_template:
                    inputs = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        return_tensors='pt', return_dict=True)
                else:
                    prompt = ''.join(f"{m['role'].upper()}: {m['content']}\n" for m in messages) + 'ASSISTANT:'
                    inputs = tokenizer(prompt, return_tensors='pt')
                token_count = int(inputs['input_ids'].shape[-1])
                limit = int(cfg.get('context', 4096))
                allocated = min(max_tokens, limit - token_count - 16)
                if allocated < 64:
                    raise RuntimeError(f'Prompt uses {token_count} tokens; context of {limit} leaves '
                                       'insufficient room for a response. Reduce the spec/peer excerpts '
                                       'or increase this model\'s configured context.')
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                options = {'max_new_tokens': allocated, 'do_sample': temperature > 0}
                if temperature > 0:
                    options['temperature'] = max(0.01, temperature)
                if tokenizer.pad_token_id is not None:
                    options['pad_token_id'] = tokenizer.pad_token_id
                elif tokenizer.eos_token_id is not None:
                    options['pad_token_id'] = tokenizer.eos_token_id
                with torch.inference_mode():
                    result = model.generate(**inputs, **options)
                new_ids = result[0][inputs['input_ids'].shape[-1]:]
                generated = int(new_ids.shape[-1])
                endings = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
                if isinstance(tokenizer.eos_token_id, list):
                    endings = set(tokenizer.eos_token_id)
                if getattr(model.generation_config, 'eos_token_id', None) is not None:
                    eos = model.generation_config.eos_token_id
                    endings.update(eos if isinstance(eos, list) else [eos])
                ended = any(int(token) in endings for token in new_ids.tolist())
                reason = 'stop' if ended or generated < allocated else 'length'
                return {'text': tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
                        'finish_reason': reason, 'output_tokens': generated, 'applied_max_tokens': allocated}
            if kind == 'openai':
                import httpx
                base = cfg['base_url'].rstrip('/')
                endpoint = base if base.endswith('/chat/completions') else base + '/chat/completions'
                headers = {}
                key = os.environ.get(cfg.get('api_key_env', 'COUNCIL_API_KEY'), '')
                if key:
                    headers['Authorization'] = f'Bearer {key}'
                response = httpx.post(endpoint, json={
                    'model': cfg['model_name'], 'messages': messages,
                    'max_tokens': max_tokens, 'temperature': temperature
                }, headers=headers, timeout=300)
                response.raise_for_status()
                body = response.json()
                choice = body['choices'][0]
                return {'text': choice['message'].get('content') or '',
                        'finish_reason': choice.get('finish_reason') or 'unknown',
                        'output_tokens': (body.get('usage') or {}).get('completion_tokens')}
            raise ValueError('Unrecognized inference backend.')


def _demo_response(messages):
    """Explicitly simulated output used for UX smoke testing, not a language model."""
    prompt = messages[-1]['content'].lower()
    if 'return valid json only' in prompt:
        return ('{"executive_summary":"DEMO ONLY — no language model was used. '
                'This session verifies council wiring, not a real review.",'
                '"findings":[{"title":"Replace demo with an installed language model",'
                '"severity":"high","recommendation":"Install a GGUF or Transformers model '
                'and select it in agent configurations.","sources":[]}],'
                '"agreement":[],"dissent":[],"open_questions":[],"next_steps":[]}')
    return ('[SIMULATED DEMO — NOT AN AI FINDING]\n'
            'This is a placeholder response for testing the orchestration interface. '
            'Install and select real models for substantive analysis.')
