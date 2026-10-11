<#
.SYNOPSIS
  Launch the cost-reduced Erdos experiments (setups A-D, "_cheap" variants).

.DESCRIPTION
  1. Points ERDOS_WARM_START at the warm-start file (seeds read it from there,
     so it is not pasted into every model prompt).
  2. Runs a ~1-cent preflight: refuses to start if the Anthropic key is
     invalid or the credit balance is empty (so runs cannot silently stall).
  3. Starts each requested setup as a hidden background process, logging to
     cheap_run\logs\.

.EXAMPLE
  .\cheap_run\run_cheap.ps1                 # A and D only (best value, max $4)
  .\cheap_run\run_cheap.ps1 -Setups A,B,D   # add B (max $2 more)
  .\cheap_run\run_cheap.ps1 -Setups A,B,C,D # everything (max $7 total)
#>
param(
    [ValidateSet('A', 'B', 'C', 'D')]
    [string[]]$Setups = @('A', 'D'),
    [switch]$SkipPreflight   # for offline testing only: gen-0 needs no API call
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$exe = Join-Path $Repo '.venv\Scripts\shinka_launch.exe'
$py  = Join-Path $Repo '.venv\Scripts\python.exe'
$npy = Join-Path $Repo 'examples\erdos_continuous\warm_start_n600.npy'

if (-not (Test-Path $exe)) { throw "Missing $exe - set up the virtual environment first (see cheap_run\README.md)." }
if (-not (Test-Path $npy)) { throw "Missing warm start file: $npy" }
if (-not (Test-Path (Join-Path $Repo '.env'))) { throw "Missing .env in $Repo - copy cheap_run\.env.example to .env and fill in your keys." }

$env:ERDOS_WARM_START = $npy
$env:PYTHONIOENCODING = 'utf-8'

if (-not $SkipPreflight) {
    Write-Host 'Preflight: checking API keys and credit (costs about a cent)...'
    & $py (Join-Path $PSScriptRoot 'preflight.py')
    if ($LASTEXITCODE -ne 0) { throw 'Preflight failed - fix the problem above, then run again.' }
}

$logs = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

foreach ($s in $Setups) {
    $variant = 'erdos_setup_' + $s.ToLower() + '_cheap'
    $log = Join-Path $logs ($variant + '_' + $stamp + '.log')
    $p = Start-Process -FilePath $exe -ArgumentList ('variant=' + $variant) `
        -WorkingDirectory $Repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
    Write-Host ("Started setup {0} (PID {1}) -> {2}" -f $s, $p.Id, $log)
}

Write-Host ''
Write-Host 'Check progress any time with:   .venv\Scripts\python.exe cheap_run\status.py'
Write-Host 'Stop everything with:           .\cheap_run\stop_cheap.ps1'
