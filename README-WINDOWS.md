# strata-nemotron for Windows (unofficial build)

Windows x64 build of [strata-nemotron](https://github.com/Cudila/Strata/tree/strata-nemotron), the engine that runs
[Nemotron-3.5-Super-120B-A12B NVFP4](https://huggingface.co/Cudecnik/Nemotron-3.5-Super-120B-A12B-NVFP4-Strata-Blackwell)
on one RTX 50-series card. Upstream builds and tests on Linux only; this build adds the Windows patches from
branch [`windows-nemotron`](https://github.com/dseditor/Strata/tree/windows-nemotron) of the fork.

## Requirements

- **RTX 50-series GPU** (sm_120). RTX 40/30 and AMD will not run it.
- NVIDIA driver **580 or newer** (CUDA 13).
- **96 GB RAM** recommended (the experts take ~60 GB of page-locked RAM; 64 GB is not enough).
- Python 3.10+ (for the OpenAI-compatible server).
- About 72 GB of disk for the model, 1.7 GB more for images.

## Install and run

1. Download **`strata-nemotron-windows-x64.zip`** from the
   [latest release](https://github.com/dseditor/Strata/releases/latest) and unzip it (a drive with ~75 GB free).
2. Double-click **`START-NEMOTRON.bat`** in the unzipped folder.

(From a git clone of branch `windows-nemotron` it works the same: the installer downloads the engine part of the zip.)

The first run:
1. checks the PC (RTX 50-series, driver 580+, RAM, free disk) and stops before downloading anything if it does not fit;
2. installs Python 3.12 for your user account if there is none, and a private `.venv` in this folder;
3. downloads the Windows engine from the [release](https://github.com/dseditor/Strata/releases/tag/windows-nemotron-v1) (450 MB);
4. asks whether you want image input, then downloads the model from Hugging Face into `models\` (72 GB, +1.7 GB for
   images; resumable: if it stops, double-click again);
5. extracts the tokenizer and starts the server, then opens http://127.0.0.1:8080/ when the model is loaded.

Later runs skip everything that is done and just start the model (1-4 minutes to load).
OpenAI clients: `http://127.0.0.1:8080/v1`. Options: `START-NEMOTRON.bat --no-images`, `--port N`, `--no-open`.

Manual install instead: unzip the release into this folder, put the shards (and `mmproj-BF16.gguf`) into `models\`,
and run `tools\windowsun-nemotron.bat`.

## Measured (RTX 5090 32 GB, 96 GB DDR5-4800, i5-13500, Windows 11)

| | |
|---|---|
| engine bench, 4K prompt | prefill 1,475 tok/s, decode 96 tok/s (MTP, fast mode), expert hit 96% |
| through the server | Chinese chat 77 tok/s; a 3,369-token photo described correctly in 19 s |
| first load | 211 s from disk |

## Windows patches

1. `IN` / `OUT` macros from the Windows headers clash with a local name in `nemo_model.cpp`.
2. `unistd.h` -> `io.h`, `setenv` -> `_putenv_s`, no `SIGPIPE`.
3. Image rows: Windows caps page-locked RAM well below the RAM size and the experts use it up, so the image buffer falls
   back to pageable memory instead of failing with "no page-locked memory for the image rows".

Built with VS 2022 Build Tools (MSVC 14.42) + CUDA 13.4, `STRATA_PORTABLE=ON` (AVX2 baseline, static MSVC runtime).
`cublas64_13.dll` and `cublasLt64_13.dll` are NVIDIA CUDA redistributables.

Engine: MIT (Strata contributors). Model: OpenMDW-1.1 (see the model card).
