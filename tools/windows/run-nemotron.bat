@echo off
REM Start Nemotron-3.5-Super (strata-nemotron) with the OpenAI/Anthropic-compatible server on Windows.
REM Run from the repository root. Needs: RTX 50-series GPU, driver 580+, 96 GB RAM, Python 3.10+.
REM Put both model shards (and mmproj-BF16.gguf for images) in the models folder first.
setlocal
cd /d "%~dp0..\.."
set "PATH=%CD%\build;%PATH%"
if not exist build\strata-nemotron.exe (
  echo build\strata-nemotron.exe not found. Unzip the Windows release here, or build with tools\windows\build-nemotron.bat.
  exit /b 1
)
if not exist models\Nemotron-3.5-Super-120B-A12B-NVFP4-00001-of-00002.gguf (
  echo Model not found. Download both shards into the models folder:
  echo https://huggingface.co/Cudecnik/Nemotron-3.5-Super-120B-A12B-NVFP4-Strata-Blackwell
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo Creating Python environment .venv ...
  python -m venv .venv || exit /b 1
  .venv\Scripts\python.exe -m pip install -r requirements.txt || exit /b 1
)
if not exist models\tokenizer\vocab.json (
  echo Extracting the tokenizer from the GGUF ...
  .venv\Scripts\python.exe tools\strata_tokenizer.py --gguf models\Nemotron-3.5-Super-120B-A12B-NVFP4-00001-of-00002.gguf --out models || exit /b 1
)
set "CONFIG=docs\nemotron\strata-nemotron.windows.json"
if not exist models\mmproj-BF16.gguf (
  echo mmproj-BF16.gguf not found: starting without images.
  .venv\Scripts\python.exe -c "import json;c=json.load(open(r'%CONFIG%'));c.pop('vision',None);json.dump(c,open(r'models\nemotron-text-only.json','w'),indent=2)"
  set "CONFIG=models\nemotron-text-only.json"
)
.venv\Scripts\python.exe serve\server.py --engine strata --config %CONFIG% --port 8080
endlocal
