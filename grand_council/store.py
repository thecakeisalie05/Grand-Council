"""JSON persistence. Writes are atomic; secrets are intentionally not persisted."""
import json
import os
from pathlib import Path
from threading import RLock


class Store:
    def __init__(self, base):
        self.base = Path(base).expanduser().resolve()
        self.base.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()

    def get(self, name, default):
        with self.lock:
            f = self.base / f"{name}.json"
            if not f.exists():
                return default
            return json.loads(f.read_text(encoding="utf-8"))

    def put(self, name, value):
        with self.lock:
            file = self.base / f"{name}.json"
            tmp = file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fp:
                json.dump(value, fp, indent=2, ensure_ascii=False)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(tmp, file)

    def get_run(self, run_id):
        with self.lock:
            f = self.base / 'runs' / f'{run_id}.json'
            if not f.exists():
                return None
            return json.loads(f.read_text(encoding='utf-8'))

    def put_run(self, run):
        with self.lock:
            directory = self.base / 'runs'
            directory.mkdir(exist_ok=True)
            target = directory / f"{run['id']}.json"
            tmp = target.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(run, f, ensure_ascii=False, indent=2)
                f.flush()
            os.replace(tmp, target)

    def list_runs(self):
        with self.lock:
            directory = self.base / 'runs'
            if not directory.exists():
                return []
            out = []
            for f in directory.glob('*.json'):
                try:
                    run = json.loads(f.read_text('utf-8'))
                    out.append({k: run.get(k) for k in ('id', 'title', 'status', 'created_at', 'updated_at', 'rounds', 'phase', 'error')})
                except (OSError, ValueError):
                    continue
            return sorted(out, key=lambda v: v.get('created_at') or '', reverse=True)
