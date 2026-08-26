$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$EnvDir = Join-Path $PSScriptRoot ".venv"
$Python = Join-Path $EnvDir "Scripts\python.exe"

Push-Location $Root
try {
    uv python install 3.11
    uv venv $EnvDir --python 3.11
    & $Python -m ensurepip --upgrade
    & $Python -m pip install --upgrade pip setuptools wheel
    & $Python -m pip install -r (Join-Path $PSScriptRoot "requirements-tts.txt")
    & $Python -c "import sys, torch, TTS; print('python', sys.version.split()[0]); print('TTS', TTS.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}
finally {
    Pop-Location
}
