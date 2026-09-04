$ErrorActionPreference = "Stop"

$Package = "deepfilternet-rs"
$Version = "0.1.1"
$ToolRoot = Join-Path $env:LOCALAPPDATA "LYKENOX-external-tools\deepfilternet-rs-$Version"
$ToolPython = Join-Path $ToolRoot "Scripts\python.exe"
$ToolExe = Join-Path $ToolRoot "Scripts\deepfilternet.exe"

function Test-CompatiblePython([string]$PythonExe) {
    if (-not (Test-Path $PythonExe)) { return $false }
    $ok = & $PythonExe -c "import sys; print('yes' if sys.version_info[:2] >= (3,10) and sys.version_info[:2] <= (3,12) else 'no')"
    return ($LASTEXITCODE -eq 0 -and $ok.Trim() -eq "yes")
}

$Bootstrap = $null
$ProjectPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (Test-CompatiblePython $ProjectPython) {
    $Bootstrap = @($ProjectPython)
} else {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $py) {
        foreach ($minor in @("3.12", "3.11", "3.10")) {
            & py "-$minor" -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $Bootstrap = @("py", "-$minor")
                break
            }
        }
    }
}

if ($null -eq $Bootstrap) {
    throw "DeepFilterNet-rs requires Python 3.10-3.12. Install Python 3.12 (x64) and rerun this script."
}

New-Item -ItemType Directory -Force -Path (Split-Path $ToolRoot -Parent) | Out-Null
if (-not (Test-Path $ToolPython)) {
    if ($Bootstrap.Count -eq 1) {
        & $Bootstrap[0] -m venv $ToolRoot
    } else {
        & $Bootstrap[0] $Bootstrap[1] -m venv $ToolRoot
    }
    if ($LASTEXITCODE -ne 0) { throw "Failed to create isolated external tool environment." }
}

& $ToolPython -m pip install --disable-pip-version-check --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip in external tool environment." }

& $ToolPython -m pip install --disable-pip-version-check "$Package==$Version"
if ($LASTEXITCODE -ne 0) { throw "Failed to install $Package==$Version in external tool environment." }

if (-not (Test-Path $ToolExe)) {
    throw "Installation completed but expected executable was not found: $ToolExe"
}

$Installed = (& $ToolPython -m pip show $Package | Select-String '^Version:').ToString().Split(':', 2)[1].Trim()
Write-Host "external_tool_root=$ToolRoot"
Write-Host "external_tool_package=$Package"
Write-Host "external_tool_version=$Installed"
Write-Host "deepfilternet_exe=$ToolExe"
Write-Host "integrated_into_lykenox=false"
Write-Host "lykenox_project_venv_modified=false"
