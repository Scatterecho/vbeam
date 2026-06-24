param(
    [int]$Port = 8899
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$envPython = "C:\Users\Admin\.conda\envs\vbeam_jax_cpu\python.exe"
$jupyterBase = Join-Path $env:TEMP "vbeam_jupyter_$PID"

$env:PYTHONNOUSERSITE = "1"
$env:JUPYTER_CONFIG_DIR = Join-Path $jupyterBase "config"
$env:JUPYTER_DATA_DIR = Join-Path $jupyterBase "data"
$env:JUPYTER_RUNTIME_DIR = Join-Path $jupyterBase "runtime"
$env:IPYTHONDIR = Join-Path $jupyterBase "ipython"
$env:JUPYTER_PATH = "C:\Users\Admin\.conda\envs\vbeam_jax_cpu\share\jupyter"
$env:JUPYTER_PLATFORM_DIRS = "1"

New-Item -ItemType Directory -Force -Path $env:JUPYTER_CONFIG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:JUPYTER_DATA_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:JUPYTER_RUNTIME_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:IPYTHONDIR | Out-Null

Set-Location $repoRoot
& $envPython -s -m jupyterlab --no-browser --port=$Port --ServerApp.root_dir="$repoRoot"
