@echo off
REM Build strata-nemotron on Windows (MSVC + CUDA 13, RTX 50-series / sm_120)
REM Requires: Visual Studio 2022 (or Build Tools) with C++ workload, CUDA Toolkit 13.x, cmake, ninja
REM Usage: run from the repository root. Set VCVARS / CUDA_PATH below if your paths differ.
setlocal
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo vcvars64.bat not found. Set VCVARS to your Visual Studio vcvars64.bat path.
  exit /b 1
)
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
  echo nvcc.exe not found under %CUDA_PATH%. Install CUDA Toolkit 13.x or set CUDA_PATH.
  exit /b 1
)
call "%VCVARS%" >nul
set "CUDACXX=%CUDA_PATH%\bin\nvcc.exe"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_PORTABLE=ON || exit /b 1
cmake --build build -j --target strata-nemotron || exit /b 2
echo.
echo Built: build\strata-nemotron.exe
echo At run time put "%CUDA_PATH%\bin\x64" on PATH (cublas64_13.dll, cudart64_13.dll).
endlocal
