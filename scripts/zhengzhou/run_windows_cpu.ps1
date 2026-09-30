param(
    [string]$Python = 'D:\model_coupling_generated_20260930\cpu_env\Scripts\python.exe',
    [Parameter(Mandatory=$true)][string[]]$PythonArgs
)
$ErrorActionPreference = 'Stop'
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python environment missing: $Python" }
$taskPackage = & $Python -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("rasterio").origin).parent)'
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate project rasterio installation' }
$taskProj = Join-Path $taskPackage 'proj_data'
$taskGdal = Join-Path $taskPackage 'gdal_data'
if (!(Test-Path -LiteralPath (Join-Path $taskProj 'proj.db'))) { throw 'Bundled PROJ database missing' }
$taskPrevious = @{}
foreach ($taskKey in @('PROJ_LIB', 'PROJ_DATA', 'GDAL_DATA')) { $taskPrevious[$taskKey] = [Environment]::GetEnvironmentVariable($taskKey, 'Process') }
try {
    $env:PROJ_LIB = $taskProj
    $env:PROJ_DATA = $taskProj
    $env:GDAL_DATA = $taskGdal
    & $Python @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Project Python command failed (exit $LASTEXITCODE)" }
} finally {
    foreach ($taskKey in $taskPrevious.Keys) { [Environment]::SetEnvironmentVariable($taskKey, $taskPrevious[$taskKey], 'Process') }
}
