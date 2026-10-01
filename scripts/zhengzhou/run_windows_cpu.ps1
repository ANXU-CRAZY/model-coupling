param(
    [string]$Python = 'D:\model_coupling_generated_20260930\cpu_env\Scripts\python.exe',
    [Parameter(Mandatory=$true)][string[]]$PythonArgs
)
$ErrorActionPreference = 'Stop'
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python environment missing: $Python" }
$taskPackage = & $Python -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('rasterio').origin).parent)"
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate project rasterio installation' }
$taskProj = Join-Path $taskPackage 'proj_data'
$taskGdal = Join-Path $taskPackage 'gdal_data'
if (!(Test-Path -LiteralPath (Join-Path $taskProj 'proj.db'))) { throw 'Bundled PROJ database missing' }
$taskPrevious = @{}
$taskKeys = @('PROJ_LIB', 'PROJ_DATA', 'GDAL_DATA', 'GIT_CONFIG_COUNT', 'PYTHONPATH')
foreach ($taskKey in $taskKeys) { $taskPrevious[$taskKey] = [Environment]::GetEnvironmentVariable($taskKey, 'Process') }
$taskGitCount = 0
if (![string]::IsNullOrWhiteSpace($env:GIT_CONFIG_COUNT)) { $taskGitCount = [int]$env:GIT_CONFIG_COUNT }
$taskGitKeyName = "GIT_CONFIG_KEY_$taskGitCount"
$taskGitValueName = "GIT_CONFIG_VALUE_$taskGitCount"
$taskPrevious[$taskGitKeyName] = [Environment]::GetEnvironmentVariable($taskGitKeyName, 'Process')
$taskPrevious[$taskGitValueName] = [Environment]::GetEnvironmentVariable($taskGitValueName, 'Process')
$taskProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
try {
    $env:PROJ_LIB = $taskProj
    $env:PROJ_DATA = $taskProj
    $env:GDAL_DATA = $taskGdal
    [Environment]::SetEnvironmentVariable($taskGitKeyName, 'safe.directory', 'Process')
    [Environment]::SetEnvironmentVariable($taskGitValueName, $taskProjectRoot.Replace('\', '/'), 'Process')
    $env:GIT_CONFIG_COUNT = [string]($taskGitCount + 1)
    if ([string]::IsNullOrWhiteSpace($taskPrevious['PYTHONPATH'])) {
        $env:PYTHONPATH = $taskProjectRoot
    } else {
        $env:PYTHONPATH = $taskProjectRoot + [System.IO.Path]::PathSeparator + $taskPrevious['PYTHONPATH']
    }
    & $Python @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Project Python command failed (exit $LASTEXITCODE)" }
} finally {
    foreach ($taskKey in $taskPrevious.Keys) { [Environment]::SetEnvironmentVariable($taskKey, $taskPrevious[$taskKey], 'Process') }
}
