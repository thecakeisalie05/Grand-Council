"""Safe, predictable runtime pip plans and disk-space diagnostics.

Pip runtime installation is explicitly opted into by the user. All arguments
are allowlisted; model IDs, paths or other user strings never become pip args.
"""
from pathlib import Path
import shutil
import tempfile

GIB = 1024 ** 3
CPU_WHEEL_INDEX = 'https://download.pytorch.org/whl/cpu'

# These are conservative *preflight minimums*, not exact size predictions.
MINIMUM_FREE_GB = {'gguf': 1.5, 'transformers-cpu': 3.0, 'transformers': 7.0}
TRANSFORMERS_PACKAGES = [
    'transformers>=4.48,<6', 'accelerate>=1.5', 'safetensors>=0.4',
    'sentencepiece>=0.2', 'protobuf>=5',
]


def install_commands(kind, python_executable):
    """Return pip commands without shell interpolation or wheel caches."""
    base = [python_executable, '-m', 'pip', 'install', '--no-cache-dir', '--prefer-binary']
    if kind == 'gguf':
        return [base + ['llama-cpp-python>=0.3']]
    if kind == 'transformers-cpu':
        # Use official CPU wheels for torch. Installing the other packages from
        # PyPI in a separate step avoids restricting their index to PyTorch.
        return [base + ['--index-url', CPU_WHEEL_INDEX, 'torch>=2.2'],
                base + TRANSFORMERS_PACKAGES]
    if kind == 'transformers':
        # Keeps the original installation path for people who need the standard
        # torch build; CUDA compatibility still depends on hardware and wheel.
        return [base + ['torch>=2.2'] + TRANSFORMERS_PACKAGES]
    raise ValueError('Choose gguf, transformers-cpu, or transformers.')


def storage_report(python_prefix, temp_dir=None, model_dir=None):
    """Report separate volumes for installation, temporary files and models."""
    locations = {'python_environment': Path(python_prefix),
                 'temporary_files': Path(temp_dir or tempfile.gettempdir())}
    if model_dir is not None:
        locations['model_library'] = Path(model_dir)
    report = {}
    for label, path in locations.items():
        candidate = path.resolve()
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        try:
            usage = shutil.disk_usage(candidate)
            report[label] = {'path': str(path), 'free_gb': round(usage.free / GIB, 2),
                             'total_gb': round(usage.total / GIB, 2)}
        except OSError:
            report[label] = {'path': str(path), 'free_gb': None, 'total_gb': None}
    return report


def require_disk_room(kind, report):
    minimum = MINIMUM_FREE_GB[kind]
    for key in ('python_environment', 'temporary_files'):
        location = report.get(key, {})
        free = location.get('free_gb')
        if free is not None and free < minimum:
            name = 'Python environment' if key == 'python_environment' else 'Windows temporary files'
            raise RuntimeError(
                f'Not enough disk space for {kind}: {name} has only {free:.2f} GB free. '
                f'Allow at least {minimum:.1f} GB free on that drive before attempting this install '
                '(actual requirements may be higher). You can reclaim pip cache space with '
                '".venv\\Scripts\\python.exe -m pip cache purge", or use a GGUF model / local model server '
                'without PyTorch. Changing TMP alone will not help if the Python environment drive is full.'
            )


def pip_failure_message(exit_code, tail):
    output = '; '.join(tail)[-800:]
    low = output.lower()
    if any(x in low for x in ('no space left on device', '[errno 28]', 'winerror 112',
                              'not enough space on the disk', 'there is not enough space on the disk')):
        return (f'Python runtime installation ran out of disk space (pip exit {exit_code}). '
                'Free space on the Python environment and temporary-file drives; '
                'try ".venv\\Scripts\\python.exe -m pip cache purge", '
                'then select Transformers CPU-only (smaller) or GGUF. '
                'Do not delete the whole .venv unless you intend to reinstall the application.')
    return f'pip exited with code {exit_code}: {output}'
