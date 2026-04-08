param(
    [switch]$SkipMkDocs,
    [switch]$SkipSphinx
)

$ErrorActionPreference = "Stop"

# Avoid treating non-fatal stderr text from native tools as PowerShell errors.
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$FailMessage
    )

    $stdoutFile = [System.IO.Path]::GetTempFileName()
    $stderrFile = [System.IO.Path]::GetTempFileName()
    try {
        $process = Start-Process -FilePath $Exe -ArgumentList $Arguments -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
        $exitCode = $process.ExitCode

        if (Test-Path $stdoutFile) {
            Get-Content $stdoutFile | ForEach-Object { $_ }
        }
        if (Test-Path $stderrFile) {
            Get-Content $stderrFile | ForEach-Object { $_ }
        }
    }
    finally {
        if (Test-Path $stdoutFile) { Remove-Item $stdoutFile -Force }
        if (Test-Path $stderrFile) { Remove-Item $stderrFile -Force }
    }

    if ($exitCode -ne 0) {
        throw "$FailMessage (exit code $exitCode)."
    }
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$preferredPython = "d:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Env\entorno_CAMP10\Scripts\python.exe"
if (Test-Path $preferredPython) {
    $pythonExe = $preferredPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = "python"
} else {
    throw "No Python executable found. Install Python or update `$preferredPython in build_docs.ps1."
}

$autoApiCache = Join-Path $root "sphinx_docs\source\autoapi"
if (Test-Path $autoApiCache) {
    Remove-Item $autoApiCache -Recurse -Force
}

if (-not $SkipSphinx) {
    Invoke-NativeCommand `
        -Exe $pythonExe `
        -Arguments @("-m", "sphinx", "-b", "html", "sphinx_docs/source", "docs/technical") `
        -FailMessage "Sphinx build failed"
}

if (-not $SkipMkDocs) {
    Invoke-NativeCommand `
        -Exe $pythonExe `
        -Arguments @("-m", "mkdocs", "build") `
        -FailMessage "MkDocs build failed"
}
