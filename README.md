# Grand Council

A Windows desktop and optional browser-based studio for multi-agent deliberation on large specifications. Multiple agents can use **the same Hugging Face model or different models**, each with a custom mandate, sampling temperature, and response budget.

## Included

- Modern, responsive local dashboard served by FastAPI. No cloud account, JavaScript build step, or external frontend CDN.
- Hugging Face repository inspection, gated model access via a temporary token, revision pinning, and checkpoint downloads.
- **GGUF** inference with optional `llama-cpp-python`, **safetensors causal LMs** with optional `transformers`/PyTorch, or any **OpenAI-compatible inference server** (e.g. locally hosted llama.cpp server or LM Studio). Choose an individual model per council member.
- Local memory-aware model cache (keep one resident by default; automatic least-recently-used eviction), load/unload actions, GPU layer count, context configuration, and CPU/CUDA hardware readout.
- Independent proposals → peer critiques → revised positions → chair synthesis. Configure 0–5 rounds, 2–12 agents, and evaluation priorities.
- Live progress, complete attributed transcripts, structured report (severity, sources, consensus/dissent, questions and next steps), archived sessions, Markdown/JSON export, and cancellation between inference calls.
- Explicit **demo adapter** for smoke testing; **demo messages are simulations**, not AI reviews.

## Quick start

**Windows desktop (recommended):** Download the `GrandCouncil-Windows-x64` artifact from a successful [Windows desktop workflow](https://github.com/thecakeisalie05/Grand-Council/actions/workflows/windows-desktop.yml), extract the whole ZIP, and launch `GrandCouncil.exe`. No Python install required. See the desktop section below for GPU setup.

**Local source ZIP without a published GitHub build:** Run `Publish-To-GitHub.ps1` from the extracted source folder (requires Git for Windows and your GitHub credentials) to upload the full project and trigger the Windows build. The Python launcher below remains available for development.

### Legacy browser-based source launcher

Requires **Python 3.10+**, a modern browser, and sufficient RAM/VRAM for the models you choose.

### Windows (one-click)

Double-click **`launch_windows.bat`**. On first launch it creates an isolated Python environment and installs the app. The launcher detects an existing Grand Council instance and opens it rather than crashing. If the default port `8765` is occupied by another program, it tries the next available port and opens the correct address once the server is ready. Install model runtimes from **Model Library** after the app starts.

### Windows (PowerShell)

```powershell
cd grand-council
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m grand_council.server
```

If script execution is restricted in PowerShell, use `.\.venv\Scripts\python.exe -m pip install -e .` and `.\.venv\Scripts\python.exe -m grand_council.server` instead.

### Linux or macOS

Run `./launch_linux_mac.sh` for an automatic first-time setup, or use:

```bash
cd grand-council
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m grand_council.server
```

Open **http://127.0.0.1:8765** in a browser.

### Install local inference backends

Use the Model Library's **Install GGUF runtime**, **Transformers · CPU-only (smaller)**, or **Transformers · standard PyTorch (large)** buttons, or install in your environment manually:

```bash
python -m pip install -e '.[gguf]'
# or
python -m pip install -e '.[transformers]'
```

### Disk-space errors during runtime installation

If an installation fails with `OSError: [Errno 28] No space left on device`, Windows has run out of space in the environment or temporary-files location. The installer now checks available space before starting and runs pip with `--no-cache-dir` to avoid retaining downloaded wheels. The minimum free-space checks are **not exact size forecasts**; larger builds can still require additional space.

From `grand-council`, run this to clear **pip's download cache only** (not your app or models):

```bat
.venv\Scripts\python.exe -m pip cache purge
```

Then use **Transformers · CPU-only (smaller)** if CPU inference is acceptable, or preferably **GGUF** with `llama.cpp` / an external OpenAI-compatible local server if you need low-storage quantized models. CPU-only PyTorch **cannot** use your NVIDIA GPU for Transformers inference. A standard CUDA PyTorch build can be much larger and must match the GPU's compute capability.

Check remaining storage in File Explorer or PowerShell (`Get-PSDrive C`). The default Python virtual environment is within the extracted Grand Council folder and the default model library is in `%USERPROFILE%\.grand-council`; placing the extracted app and the model library on a drive with more capacity is an option. For example, set `GRAND_COUNCIL_HOME` before starting the app to change the model-library directory. Moving `TEMP` to another drive alone cannot fix a full drive containing `.venv`.

On Windows, GGUF CPU builds may require Microsoft C++ Build Tools if a prebuilt wheel is unavailable. For NVIDIA acceleration, use a CUDA-enabled `llama-cpp-python` build matching your toolchain. A PyTorch installation with CUDA support may also need a build from PyTorch's official install selector; CPU-only PyTorch will not use your NVIDIA GPU.

**Your GTX 1080-class 8 GB GPU** is most suited to smaller/quantized GGUF language models; use one loaded model at a time initially. The app uses `n_gpu_layers=-1` by default for GGUF, requesting GPU offload where the installed backend supports it. Test with a small model before using large multi-agent configurations.

## Configure a council

1. In **Model Library**, enter a Hugging Face model repo (e.g. a GGUF repo or a Transformers-compatible text generation repo), click **Inspect**, select a `.gguf` file or Transformers backend, then **Download to local library**. Some models require accepting license terms on Hugging Face and supplying an access token. The token is not stored.
2. Wait for completion in **Activity**. Reload the page if needed. Assign the model to any agent in **Council chamber**, or use the same model for all agents.
3. Paste a spec or import a text/Markdown/JSON/YAML file. Adjust agent prompts, chair, review criteria, response detail (concise / balanced / thorough), optional automatic continuation, context budget and debate rounds. Click **Convene council**.
4. Follow agent messages in **Deliberations**, expand any response, then inspect sources in the final report. Export as Markdown or JSON. Review saved sessions in **Archives**.

### Response completion and continuation (v0.1.3)

The Council chamber now exposes **Response detail** instead of a single per-agent token count. It allocates budgets by phase, giving long proposals and chair syntheses more room than focused critiques:

| Mode | Proposals | Critiques | Revisions | Chair synthesis |
| --- | ---: | ---: | ---: | ---: |
| Concise | 768 | 512 | 768 | 1,536 |
| Balanced (default) | 1,536 | 896 | 1,280 | 3,072 |
| Thorough | 2,560 | 1,536 | 2,048 | 4,096 |

Each entry is the **maximum output tokens requested per generation**, not a guaranteed number of tokens. The local GGUF and Transformers adapters reduce the requested allowance when necessary to leave room in the configured model context window. The prompt itself can still be too long, requiring a larger context or shorter inputs. OpenAI-compatible endpoints decide their own actual context limits.

**Continue incomplete responses** is enabled by default, with up to **one additional generation** when a backend explicitly reports it stopped due to output length. You can choose zero or two additional generations. Continuing invokes the model again with the original task and its response so far; the app tries to merge overlapping text. Continuation is not perfectly seamless, and long responses may exhaust context even with adaptive budgets. The partial text is then kept and visibly flagged, never silently marked finished. An unknown provider finish reason is shown as **Finish unknown** and is not guessed to mean truncation.

The proceedings show **Completed**, **Completed · N parts**, **Potentially cut off**, or **Finish unknown** for each message. Markdown/JSON exports retain completion metadata, generation attempts, and warnings. Older saved sessions remain readable. A finished run can still contain flagged incomplete responses; review the warnings before trusting its summary.

Longer budgets and continuation calls take more time and may increase paid-endpoint costs. For small models or a 4,096-token context, use **Concise** or **Balanced**, and consider 8,192+ context where supported by the checkpoint and RAM/VRAM. Five debate rounds with four agents can involve dozens of calls, before any continuations.

The default council includes an architect, skeptic, implementer and QA/security evaluator. For four agents and one debate round, the council makes **13 inference calls**, all sequentially serialized through the local inference engine; model switching can increase latency substantially. You can raise the number of resident models if you have enough RAM/VRAM.

### External inference servers

In Model Library, register a base URL ending in `/v1` and a server-side model name. Any endpoint implementing `POST /v1/chat/completions` can participate alongside downloaded models. If auth is required, set environment variable `COUNCIL_API_KEY` before starting the app. Requests and specifications sent to a remote endpoint leave your computer. There is no built-in cost cap for paid endpoints.

## Large spec handling

The original spec is preserved **verbatim** in local session JSON and exports. Specs larger than the configured agent input character budget are fed to the chair **in successive chunks**. Their summaries are recursively condensed, then shared with the other agents. This is **lossy**: important details can disappear in summarization, and the output report displays a warning. Input budgets are character-based, not exact token counts. Increasing a model's context length and the budget can reduce summary loss, but the full debate context can still exceed the token window; the app surfaces a clear error rather than silently trimming Transformer inputs.

## Backend limitations and security

**“Arbitrary Hugging Face model” is not the same as “every repository can be run as a chat agent.”** Model repositories can contain encoders, diffusion models, audio/video models, adapters, unsupported architectures, incompatible quantization formats, weights that require gated access, or insufficient memory. This release targets *text-generating causal LMs* that run as GGUF with llama.cpp or safetensors with Transformers, plus compatible external inference servers. Multi-file GGUF shard downloading is supported; other required auxiliary formats (e.g. vision projectors) are not yet supported.

- Safetensors are required for Transformers downloads. Legacy pickled `.bin` weights are intentionally blocked.
- Arbitrary repository Python execution (`trust_remote_code`) is disabled by default. Enabling it can run malicious code with **your OS privileges**. Review publishers' source code first; isolate untrusted models using a container or virtual machine.
- A gated access token entered into the UI is passed to Hugging Face only for requests/downloads, kept in task memory until completion, and never written to app session/model files.
- All app HTTP traffic binds to `127.0.0.1` by default. Do not expose the server to public networks; it has no login and can install Python packages, read spec contents and delete app-managed model files.
- Local session data resides in `~/.grand-council/` (override with `GRAND_COUNCIL_HOME`), including specs, transcripts and downloaded models. Sessions may contain secrets from input specs. Downloads can be very large.
- Inference and model switching are synchronized to avoid simultaneous GPU overcommit; the council is not a parallel high-throughput service. **Multiple calls to the same model are not independent models**; diversity of role prompts is not statistical independence or evidence of correctness.
- Outputs are generated opinions, **not verified facts**. Cross-agent agreement should not be treated as proof; test consequential findings separately.
- Cancellation happens **between** model calls; an active generation or download cannot be preempted through the UI.

## Run tests

```bash
python -m pip install -e '.[dev]'
pytest -q
```

The integration tests run the explicitly simulated backend; they do **not** validate that a specific Hugging Face architecture, GPU backend, weight download, or external server functions on your machine.

## API

The FastAPI documentation is available at `http://127.0.0.1:8765/api/docs` by default (substitute the selected port if `8765` was occupied).

## Project layout

```
launch.py       # Windows startup conflict detection and browser launching
grand_council/
  server.py     # local API, model manager, job orchestration
  models.py     # GGUF, Transformers, OpenAI-compatible, demo adapters
  council.py    # iterative proposal / critique / revision / synthesis
  store.py      # atomic persistent JSON storage
  static/       # self-contained HTML / CSS / JS dashboard
tests/          # API and orchestration smoke tests
```

MIT license, see `LICENSE`.

Release 0.1.3 adds phase-specific response budgets, backend finish-reason tracking, bounded continuation, context-aware local output caps, and completion state in the transcript and exports. Release 0.1.2 added runtime disk-space preflight, CPU-only PyTorch install choice, cache-free pip installs, and more actionable storage errors.

---

## Grand Council v0.2 — native Windows desktop / automatic GGUF acceleration

The new entry point is **Grand Council Desktop**, a Windows application window
containing the existing council interface and running its local FastAPI service
inside the process. Its release archive includes Python and QtWebEngine; the
end user does **not** need to install Python, pip, CUDA Toolkit or a compiler.

1. Open **Actions → Windows desktop** in the repository and download the
   `GrandCouncil-Windows-x64` artifact from a successful run (or download the
   ZIP attached to a tagged release). Extract the complete folder; **do not**
   move just `GrandCouncil.exe` out of it.
2. Double-click `GrandCouncil.exe`. Existing records in `%USERPROFILE%\.grand-council`
   are reused. No session migration is required.
3. Open **Model library → Set up accelerated inference**. It detects your GPU
   and downloads **official llama.cpp release binaries** into the persistent
   model-data directory. Downloads are verified against the SHA256 digests in
   the GitHub release asset metadata. Alternatively, loading the first GGUF
   model will initiate the same setup automatically.
4. Inspect a Hugging Face GGUF repository, select a **Q4_K_M** or another
   quantization that fits available memory, download, and assign it to agents.
   Only one model is loaded at a time by default; large concurrent model pools
   can exhaust VRAM.

### Acceleration policy

- NVIDIA with a compatible driver: CUDA 12.4 prebuilt binary (including Pascal
  GTX 10xx, which is **not** supported by CUDA 13). If it cannot start, try
  Vulkan, then CPU. CUDA builds include the required CUDA runtime libraries.
- AMD/Intel display GPU: Vulkan, then CPU.
- Unsupported GPU or no driver: CPU. You can still use external
  OpenAI-compatible inference servers.

Setup is **automatic**, but binary downloads require an internet connection
and enough free disk space. NVIDIA display drivers and the Vulkan loader must
already be functional. The app does not silently install system-level display
GPU drivers or require administrator privileges. Performance depends on model
quantization, context size, VRAM, and the driver's capability; a GTX 1080
should be tested with a 3B–7B Q4 model and conservative context at first.

**Scope:** Official GGUF models supported by llama.cpp are the recommended
no-configuration local GPU option. Not every Hugging Face model is compatible:
PyTorch/Transformers-only safetensors models, multimodal adapters, and custom
architectures may require an external server or the developer-mode Python
runtime. The desktop binary intentionally does **not** pip-install large
third-party Python environments at runtime.

### Developer setup and Windows build

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,desktop]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe desktop_entry.py
```

The `.github/workflows/windows-desktop.yml` job tests, bundles and uploads a
portable Windows x64 ZIP on pushes and PRs; `v*` tags also publish a GitHub
release. Windows binary packaging happens in GitHub Actions, not on Linux.
The artifact is unsigned, so Windows SmartScreen may warn until a code-signing
certificate is configured. If you develop on Linux/macOS, the original
browser-based server remains usable via `grand-council`.

### Diagnostics

- `GET /api/hardware`: detected display adapters and NVIDIA driver, selected
  acceleration preference, installed backends, and free disk space.
- `GET /api/jobs`: installation/model download progress and errors.
- `%USERPROFILE%\.grand-council\logs\llama-server-<port>.log`: local
  inference process logs if a backend fails to start.
- `grand-council-desktop --self-test` (developer entry point): check bundled
  package resources without launching the UI.

The server process and its local model processes are shut down when the desktop
window closes. Sessions are written atomically to the existing data folder.
