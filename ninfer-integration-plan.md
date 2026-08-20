# Plan: Integrate NInfer + Qwen3.8-27B NVFP4 into LLM Model Loader

## Goal

Add a first-class, repo-conventional way to install, register, launch, monitor, and unload the **NInfer** runtime (native `Neroued/ninfer`, model `Qwen3.8-27B NVFP4`, config from the ChatGPT write-up) alongside the existing llama.cpp flow — without breaking the current llama.cpp models/scripts/runs.

## Reality check / key constraint

- NInfer is **Linux-only**, built for **sm_120a (RTX 5090)** with CUDA 13.1, and must run inside **WSL2 Ubuntu 24.04**.
- The loader backend is a **Windows** Python/FastAPI process. It launches servers as native subprocesses (`llama-server.exe`), parses PowerShell launch scripts, and health-checks `http://host:port/health`.
- The good news: the loader's control surface is already *generic* (models → scripts → runs → health check). NInfer only needs the launch/parse/stop plumbing to understand a **`wsl.exe`-based script** and to skip llama.cpp-specific flag injection.

## Recommended approach (Option A: external runtime + loader registration)

Keep NInfer as a self-contained external runtime installed under the WSL home (`~/ninfer-qwen38`, native ext4 — building under `/mnt/c` on the 9p filesystem is very slow). The repo contributes:

1. A repo-native setup wrapper (`.cmd` + `scripts/*.ps1`) adapted from the ChatGPT one-shot installer.
2. A backend registration module that records the model + a launch preset in the loader's SQLite DB.
3. Small, targeted extensions to `scripts.py` / `runs.py` so the loader can start/stop/watch a WSL-backed NInfer server like any other model.
4. Tests + README.

Rejected: **Option B (deep integration)** — re-implementing the build/download/verify inside the loader's own backend. Duplicates the ChatGPT script, fights the loader's Windows-side download machinery, and adds long-term maintenance for no functional gain.

## Files to change

### New files

| File | Purpose |
| --- | --- |
| `setup_ninfer_qwen38.cmd` | Mirror of existing `.cmd` wrappers; invokes `scripts\setup_ninfer_qwen38.ps1`. |
| `scripts/setup_ninfer_qwen38.ps1` | Repo-styled installer (adapted from the ChatGPT script, see "Setup script changes" below). |
| `backend/app/ninfer_setup.py` | Registers the NInfer model + preset in the DB (pattern: `bonsai_setup.py` / `qwen38_setup.py`). |
| `tests/test_ninfer_setup.py` | Registration idempotency + field assertions. |
| `tests/test_ninfer_scripts.py` | Parsing of WSL/NInfer scripts. |
| `tests/test_ninfer_runs.py` | Launch-plan/stop behavior for NInfer runs. |

### Edited files

| File | Change |
| --- | --- |
| `backend/app/scripts.py` | Add `runtime` (and `wsl_distro`/`wsl_launcher`) to `ScriptInfo`; recognize `wsl.exe`/`wsl`/`ninfer-serve` as executable tokens; extract NInfer host/port/ctx/concurrency from `NINFER_*` env vars embedded in the `bash -lc` string. |
| `backend/app/runs.py` | Branch `_launch_plan` and `stop` for `runtime == "ninfer"`: resolve `wsl.exe` via `shutil.which`, skip `_with_loader_defaults()`, bypass the VRAM gate (NInfer auto-sizes its own KV pool), and on unload run `wsl.exe … bash -lc '<launcher> stop'` before killing the Windows-side process. Generalize a couple of "llama.cpp" log strings to "server". |
| `README.md` | New section: requirements, install, run, and operational notes. |

Optional (cosmetic, Phase 2): `frontend/src/main.jsx` label generalization ("llama.cpp control surface" → "Runtime control surface", "Path to llama-server.exe" → generic runtime path). Scripts/runs are already rendered generically, so the NInfer preset works in the UI without frontend changes.

## Design details

### 1. Setup script (`scripts/setup_ninfer_qwen38.ps1`)

Borrow the ChatGPT script's mechanics wholesale — they are sound:

- Discover an Ubuntu 24.04 WSL distro (`wsl.exe -l -q` + `/etc/os-release` check); optional `-Distro` param to pin it.
- Write the Bash payload to a temp file as **UTF-8 without BOM** and hand it to WSL as a literal file (`wslpath` + `wsl.exe -d <distro> -- bash <path>`) — no nested `bash -c` quoting.
- Bash payload keeps the existing preflight (nvidia-smi visible, RTX 5090 `sm_120a`, driver CUDA ≥ 13.1), apt deps (cmake, FFmpeg dev libs, libcurl ≥ 7.85), **CUDA Toolkit 13.1** install, pinned `git clone` of `Neroued/ninfer` with the `MIN_RUNTIME_REV` ancestor gate, Ninja build, HF CLI in its own venv, and the model download with SHA-256 verification.

Repo-convention changes vs. the ChatGPT script:

- **Do not `exec "$LAUNCHER"` at the end.** The loader owns start/stop. Add a `-Start` switch for direct CLI use.
- The generated launcher `~/ninfer-qwen38/run-qwen38-nvfp4.sh` gains subcommands:
  - `start` (default): unchanged ladder (262144 → 163840, floor `NINFER_MIN_CONTEXT`, default 163840), but also writes `echo $PID > "$ROOT/server.pid"`.
  - `stop`: read pidfile, `kill -INT $pid` (NInfer shuts down cleanly on SIGINT via the existing trap), remove pidfile.
  - `status`: exits 0 if the health endpoint answers.
- After the WSL run, the script captures a small facts file and writes it to the loader's app-data dir: `%LOCALAPPDATA%\llm-model-loader\ninfer-wsl.json` containing `{distro, home, launcher_path, model_path, model_name, model_sha256, model_size_bytes, runtime_revision, context_ladder, min_context, concurrency}`. This is the handoff the Python registration step reads (no fragile WSL-username guessing).
- Finally, runs the registration step:
  `python -m backend.app.ninfer_setup --info <ninfer-wsl.json>` (using the repo venv/python, same as the other setups).

Pinned facts to carry over (verify before shipping — see Open questions):

- CUDA 13.1, model `neroued/Qwen3.8-27B-nvfp4-NInfer`, file `qwen3_8_27b_nvfp4.ninfer`, ~20.02 GiB, SHA `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32`, min runtime rev `5d2c1f5590b8f4c3d106a75f65210eb4efb8f4e1`.
- Config: Vision ON, MTP3 + `--lm-head-draft`, INT8 group-64 KV, `--kv-capacity auto`, `--kv-dtype int8`, `--prefill-chunk 1024`, `--max-concurrency 3`, CUDA Graphs + prefix reuse (defaults), context ladder 262144 → 163840 with hard floor 163840.

### 2. Model registration (`backend/app/ninfer_setup.py`)

Pattern mirrors `bonsai_setup.py` / `qwen38_setup.py` (`register_*_model(model_path, runtime_path, store)`), but NInfer is not a Windows GGUF file:

- **Model row** (`source='ninfer'`, `managed=0`):
  - `path` / `normalized_path`: the WSL-side UNC reference, e.g. `\\wsl.localhost\Ubuntu-24.04\home\<user>\ninfer-qwen38\models\qwen3_8_27b_nvfp4.ninfer`. Not required to be statable on Windows.
  - `size_bytes`, `quantization='NVFP4'` (the filename `qwen3_8_27b_nvfp4.ninfer` already matches the existing `QUANT_RE` NVFP4 rule), `repo_id='neroued/Qwen3.8-27B-nvfp4-NInfer'`, `filename='qwen3_8_27b_nvfp4.ninfer'`.
  - `manual_vram_mib`: informational only (gate is bypassed for ninfer runtime) — set to a representative figure such as 30720.
  - `managed=0` so `DELETE /api/models/{id}` only removes the DB row and never touches WSL files.
  - `parse_gguf_shard('...ninfer')` returns `None` → `shard_count=1`, no consolidation impact.
- **Script row**: name e.g. `Qwen3.8-27B-NVFP4 (NInfer) / Vision / MTP3 / INT8 KV / C3 / 160kctx+`.

`raw_script` format (PowerShell, this is what `parse_script` must handle):

```powershell
& wsl.exe -d "Ubuntu-24.04" -- bash -lc 'NINFER_CONCURRENCY=3 NINFER_MIN_CONTEXT=163840 ~/ninfer-qwen38/run-qwen38-nvfp4.sh'
```

Registration is idempotent (same `normalized_path` + same preset name → update in place), exactly like the existing setup modules. Missing `ninfer-wsl.json` → clear error telling the user to run `setup_ninfer_qwen38.cmd` first.

### 3. Script parsing (`backend/app/scripts.py`)

- Add to `ScriptInfo`: `runtime: str = "llama.cpp"`, `wsl_distro: str | None = None`, `wsl_launcher: str | None = None`.
- `_is_executable_token`: also accept `wsl.exe`, `wsl`, `ninfer-serve` → `runtime = "ninfer"` when the executable is `wsl`/`wsl.exe` (or when `--ninfer`/launcher marker present for `ninfer-serve`).
- Extraction (regex-based, no subprocess):
  - distro = arg after `-d` / `--distribution`.
  - From the `-lc` string: `NINFER_HOST=` (default `127.0.0.1`), `NINFER_PORT=` (default `8081`), `NINFER_CONCURRENCY=`, `NINFER_MAX_CONTEXT=` else `NINFER_MIN_CONTEXT=` (→ `ctx_size`; `163840` in the preset, so the UI shows `160kctx`).
  - `quantization`: existing `QUANT_RE` already catches `NVFP4` from `qwen3_8_27b_nvfp4.ninfer`.
- `_powershell_to_argv` already survives this shape: backtick stripped, `& ` removed, the single-quoted `-lc` payload stays one arg after `_strip_shell_quotes`, forward slashes unaffected.
- `autosuggest_name` then works unchanged (ctx + quant + MTP flags all derive correctly).

### 4. Run manager (`backend/app/runs.py`)

In `_launch_plan`, when `parsed.get("runtime") == "ninfer"`:

- **Executable**: `wsl_path = shutil.which("wsl.exe")` (System32, always on PATH); error if missing.
- **Args**: `[wsl_path] + parsed["args"]` (`-d <distro> -- bash -lc '<env> <launcher>'`). **Do not** call `_with_loader_defaults()` — the llama.cpp `--host 0.0.0.0` / `--perf` / `--ui-config` injection would be passed to `wsl.exe` and break startup.
- **VRAM gate**: bypass with reason `"VRAM is managed by NInfer's auto-sized KV pool."` (NInfer deliberately consumes whatever fits after model+Vision+MTP+workspace+graphs; the loader's estimate is meaningless here). Keep the gate for llama.cpp scripts untouched.
- `host` / `port` from parsed → the existing `/health` probe in `_watch` works unchanged (NInfer exposes `/health`; launcher health-checks the same URL before printing READY).
- **`stop`**: for a ninfer run, first run `wsl.exe -d <distro> -- bash -lc '~/ninfer-qwen38/run-qwen38-nvfp4.sh stop'` (sends SIGINT to the Linux-side server via the pidfile), then kill the tracked `wsl.exe` process with the existing `taskkill /T /F` fallback. This matters because killing the Windows-side `wsl.exe` client alone does **not** reliably stop the Linux-side process.
- Generalize `_watch` / `stop` status strings that hardcode "llama.cpp" to "server" (update the two existing assertions in `tests/test_model_order_and_run_history.py`).

No changes needed to `_with_loader_defaults()` itself — it is only invoked on the llama.cpp path.

### 5. README

New section: **NInfer + Qwen3.8-27B NVFP4 (WSL2)**.

- Requirements: RTX 5090 (`sm_120a`), Windows NVIDIA driver exposing CUDA ≥ 13.1 into WSL, WSL2 with Ubuntu 24.04, and `wsl.exe` available.
- Install: `setup_ninfer_qwen38.cmd` (installs CUDA 13.1 toolkit in Ubuntu, builds NInfer, downloads/verifies the NVFP4 artifact, registers the preset).
- Run: via the loader GUI (preset start/stop), or directly with `~/ninfer-qwen38/run-qwen38-nvfp4.sh`; endpoint `http://127.0.0.1:8081/v1` (WSL2 forwards localhost). Port 8081 keeps NInfer distinct from llama.cpp presets on 8080.
- Notes: context ladder settles at the largest context ≥ 163840 that starts with Vision+MTP3+C3; the config never silently disables Vision/MTP/switches quant; agent clients should keep `max_tokens` around 8K–16K so the shared auto-sized KV pool isn't exhausted by reserved completions; NInfer reports its full memory/KV ledger at startup.

## Test plan

Follow the repo's `unittest` + temp-`Store` convention.

- `test_ninfer_setup.py`: idempotent registration; model row `source='ninfer'`, `managed=0`, `quantization='NVFP4'`, size from facts file; script contains `wsl.exe`, distro, launcher; `parse_script` → `runtime=='ninfer'`, `ctx_size==163840`, `port==8081`, `concurrency==3`; missing facts file raises.
- `test_ninfer_scripts.py`: `wsl.exe`/`wsl`/`ninfer-serve` recognized as executables; host/port/ctx/concurrency extracted from `-lc` env payload; llama.cpp scripts still parse to `runtime=='llama.cpp'`.
- `test_ninfer_runs.py` (monkeypatch `shutil.which`, `store.*`, `query_gpus` like the existing tests): ninfer `_launch_plan` resolves `wsl.exe`, injects **no** llama defaults, bypasses the VRAM gate with the ninfer reason; ninfer `stop` invokes the WSL stop command; existing llama tests (`_with_loader_defaults`, gated launches) keep passing.

## Out of scope / future

- **Benchmarks**: the loader's benchmark payload uses llama.cpp fields (`timings_per_token`, `reasoning_format`, `chat_template_kwargs`). NInfer is OpenAI-compatible on the surface but these fields may be ignored or rejected — make benchmark compatibility a follow-up after confirming against a live NInfer.
- Frontend runtime-browser/telemetry for WSL GPUs (Phase 2 cosmetic).
- Multi-distro settings UI.

## Open questions

1. **Keep the existing llama.cpp `setup_qwen38_nvfp4` flow?** Recommend yes — different runtime, same model family; they coexist cleanly in the DB.
2. **NInfer install root:** keep `~/ninfer-qwen38` on the WSL ext4 (recommended — native I/O for the CMake build and model) vs. the repo's `%USERPROFILE%\AI\runtimes` convention via `/mnt/c` (consistent with Bonsai but 9p-slow and worse for CUDA builds). Recommend the former.
3. **Auto-start after setup?** Recommend no (loader manages); `-Start` switch for direct use.
4. **Verify the pinned revision + SHA.** The ChatGPT-script values (`MIN_RUNTIME_REV`, model SHA) are unverified in this repo — confirm against the live `neroued/ninfer` repo and the HF artifact before shipping.
5. **VRAM gate bypass** for ninfer runtime (recommended) vs. a hard manual estimate.