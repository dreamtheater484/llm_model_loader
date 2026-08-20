# LLM Model Loader

A local browser GUI for discovering, downloading, managing, loading, unloading, and benchmarking GGUF models with `llama.cpp` — and, with one extra setup step, the native Linux NInfer runtime for Qwen3.8-27B NVFP4 over WSL2.

## Requirements

Windows 10 or 11 with `winget` and an NVIDIA GPU with a current driver. The setup script installs Python 3.12, Node.js LTS, application dependencies, and a pinned CUDA build of `llama.cpp`.

## Install

Run once from Command Prompt or PowerShell:

```bat
setup.cmd
```

The script does not start the application.

To install the CUDA-enabled PrismML runtime, download the official Ternary-Bonsai-27B Q2_0 model, and create its 128k auto-fit loading preset:

```bat
setup_bonsai_27b.cmd
```

This keeps the stock llama.cpp runtime installed by `setup.cmd` and places the Bonsai-specific runtime under `%USERPROFILE%\AI\runtimes`. On an 8 GB RTX 4070 Laptop GPU, the generated baseline preset uses an 8k Q4 KV cache so all 65 model layers and the complete KV cache remain on CUDA. Its reduced compute batch is the only additional memory accommodation; vision, speculative decoding, CPU offload, and thread tuning remain disabled. This provides a controlled all-VRAM baseline before changing one performance variable at a time.

## NInfer + Qwen3.8-27B NVFP4 (WSL2)

[NInfer](https://github.com/Neroued/ninfer) serves the `neroued/Qwen3.8-27B-nvfp4-NInfer` artifact (`qwen3_8_27b_nvfp4.ninfer`, 20 GB, SHA-256 `bb336052…81b32`) with Vision, MTP3 speculative decoding, INT8 shared KV pool, and CUDA Graphs on an RTX 5090 (`sm_120a`). It runs as a native Linux server inside WSL2; the loader starts, health-checks, and stops it like any other preset.

Requirements: an Ubuntu 24.04 WSL2 distro (name it `Ubuntu-24.04` if you want the auto-discovery to prefer it), and an NVIDIA driver that exposes CUDA 13.1+ to WSL.

```bat
setup_ninfer_qwen38.cmd
```

This installs CUDA Toolkit 13.1 and build dependencies, clones and pins NInfer (minimum runtime revision `5d2c1f5…`), builds `ninfer-serve`, downloads and SHA-verifies the model into `~/ninfer-qwen38/`, writes a persistent `run-qwen38-nvfp4.sh` launcher, and registers the "Qwen3.8-27B-NVFP4 (NInfer)" model + preset in the loader. It does not start the server. Re-running it is safe: it reuses everything already in place and only upgrades NInfer source when needed.

- **Running:** start the loader, open the model, press **Start** in *Active runs & terminal*. The launcher tries the context ladder 262144 → 163840 and keeps the largest startup-successful value at or above the 163840 floor. The loader skips its VRAM gate (NInfer auto-sizes its shared KV pool).
- **Stopping:** press **Unload** in *Active runs & terminal*. The loader signals the server inside WSL (SIGINT, then SIGKILL after 15s) and cleans up the `wsl.exe` process.
- **LAN:** like llama.cpp presets, the loader injects `NINFER_HOST=0.0.0.0` by default. `setup_ninfer_qwen38.cmd` also wires a one-time elevated portproxy + firewall rule + a `LLM-Model-Loader-NInfer-LAN` scheduled task that keeps the WSL2 NAT forwarding IP in sync; the loader re-triggers it whenever the server becomes healthy. Add `-SkipLanSetup` to the setup to stay localhost-only, or set `NINFER_HOST=127.0.0.1` in the raw script to opt out per-script.
- **Endpoint:** `http://127.0.0.1:8081/v1` (OpenAI-compatible chat completions) once loaded; request `model` id is `qwen3.8-27b`. Port 8081 keeps NInfer distinct from llama.cpp presets (which stay on 8080). The server default caps `max_tokens` at 8192 per request; stay at or below that for chat/benchmarks.
- **Bare-metal extras:** the launcher supports `~/ninfer-qwen38/run-qwen38-nvfp4.sh stop` / `status`, and its start mode forwards extra arguments to `ninfer-serve` (e.g. `--api-key`, `--max-concurrency`). All logging stays in the loader run terminal.
- **Facts:** the loader reads `ninfer-wsl.json` from `%LOCALAPPDATA%\llm-model-loader\` (or `LLM_MODEL_LOADER_DATA_DIR`) on every re-run of the setup; the registered model path is the `\\wsl.localhost\<distro>\...` UNC form.

## Run

```powershell
python run_dev.py
```

Open `http://127.0.0.1:8174`. Stop the application with `Ctrl+C`.

## Development

Run the Python tests:

```powershell
python -m unittest discover -s tests
```

Run the frontend development server separately:

```powershell
cd frontend
npm run dev
```

Settings and metadata are stored in `%LOCALAPPDATA%\llm-model-loader\loader.sqlite3`. Managed models default to `%USERPROFILE%\AI\models`, and `llama.cpp` is installed in `%USERPROFILE%\AI\llama.cpp`.


