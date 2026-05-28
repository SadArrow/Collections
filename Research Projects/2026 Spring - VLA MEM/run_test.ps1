# Run pi0.5 (pi05_droid) inference via WSL (recommended).
# Usage: from `myVLA` directory: `.\run_test.ps1`

$MyVlaDir = $PSScriptRoot
$ScriptPath = Join-Path $PSScriptRoot "run_pi05_inference.py"

function Convert-ToWslPath([string]$WindowsPath) {
    $Full = [System.IO.Path]::GetFullPath($WindowsPath)
    if ($Full.Length -lt 2 -or $Full[1] -ne ':') {
        throw "Not a drive-absolute Windows path: $Full"
    }
    $Drive = $Full.Substring(0, 1).ToLowerInvariant()
    $Rest = $Full.Substring(2).Replace('\', '/')
    return "/mnt/$Drive$Rest"
}

if (-not (Test-Path $ScriptPath)) {
    Write-Error "Script not found: $ScriptPath"
    exit 1
}

$MyVlaWsl = Convert-ToWslPath $MyVlaDir
$ScriptWsl = Convert-ToWslPath $ScriptPath
$Distro = $env:MYVLA_WSL_DISTRO

$BashCmd = "cd '$MyVlaWsl' && if command -v uv >/dev/null 2>&1; then uv run python '$ScriptWsl'; else python3 '$ScriptWsl'; fi"

if ([string]::IsNullOrWhiteSpace($Distro)) {
    & wsl -- bash -lc $BashCmd
} else {
    & wsl -d $Distro -- bash -lc $BashCmd
}

