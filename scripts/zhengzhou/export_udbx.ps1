param(
    [Parameter(Mandatory=$true)][string]$Udbx,
    [Parameter(Mandatory=$true)][string]$Out,
    [string[]]$Datasets = @('--all'),
    [string]$SuperMapHome = 'D:\SuperMap\SuperMap iDesktopX 2025',
    [string]$JdkHome = 'C:\Program Files\Java\jdk-25.0.2'
)
$ErrorActionPreference = 'Stop'
$taskInput = (Resolve-Path -LiteralPath $Udbx).Path
$taskOutput = if ([IO.Path]::IsPathRooted($Out)) { [IO.Path]::GetFullPath($Out) } else { [IO.Path]::GetFullPath((Join-Path (Get-Location).Path $Out)) }
if (Test-Path -LiteralPath $taskOutput) { throw "Output already exists: $taskOutput" }
$taskJava = Join-Path $SuperMapHome 'jre\bin\java.exe'
$taskJavac = Join-Path $JdkHome 'bin\javac.exe'
$taskDataJar = Join-Path $SuperMapHome 'bin\com.supermap.data.jar'
$taskConversionJar = Join-Path $SuperMapHome 'bin\com.supermap.data.conversion.jar'
foreach ($taskRequired in @($taskJava, $taskJavac, $taskDataJar, $taskConversionJar)) {
    if (!(Test-Path -LiteralPath $taskRequired -PathType Leaf)) { throw "Required installed SDK component missing: $taskRequired" }
}
$taskProject = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$taskBuild = Join-Path $taskProject 'local_work\sdk_build'
New-Item -ItemType Directory -Path $taskBuild -Force | Out-Null
$taskClasspath = "$taskDataJar;$taskConversionJar"
& $taskJavac --release 8 -encoding UTF-8 -cp $taskClasspath -d $taskBuild (Join-Path $PSScriptRoot 'ExportUdbx.java')
if ($LASTEXITCODE -ne 0) { throw 'Java SDK adapter compilation failed' }
$taskPreviousPath = $env:PATH
try {
    $env:PATH = (Join-Path $SuperMapHome 'bin') + ';' + $taskPreviousPath
    & $taskJava '-Dfile.encoding=UTF-8' '-Djava.awt.headless=true' "-Djava.library.path=$(Join-Path $SuperMapHome 'bin')" -cp "$taskBuild;$taskClasspath" ExportUdbx $taskInput $taskOutput @Datasets
    if ($LASTEXITCODE -ne 0) { throw 'Read-only UDBX export failed; inspect any partial output before reuse' }
} finally {
    $env:PATH = $taskPreviousPath
}
