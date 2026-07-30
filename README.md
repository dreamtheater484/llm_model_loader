# LLM Model Loader

A local browser GUI for discovering, downloading, managing, loading, unloading, and benchmarking GGUF models with `llama.cpp`.

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

