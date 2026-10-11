<#
.SYNOPSIS
  Stop the cost-reduced experiments started by run_cheap.ps1 (and only those).

.DESCRIPTION
  Matches processes whose command line contains "_cheap", so it will NOT stop
  runs of the original (non-cheap) variants. Results already written to
  results\ are kept.
#>
$targets = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'shinka_launch.exe' -and $_.CommandLine -match 'variant=erdos_setup_[abcd]_cheap'
}
if (-not $targets) {
    Write-Host 'No cost-reduced runs are running.'
    exit 0
}
foreach ($t in $targets) {
    $v = ($t.CommandLine -replace '.*variant=', 'variant=')
    Write-Host ("Stopping PID {0} ({1})" -f $t.ProcessId, $v)
    & taskkill /F /T /PID $t.ProcessId | Out-Null
}
Write-Host 'Done. Your results are kept in the results\ folder.'
