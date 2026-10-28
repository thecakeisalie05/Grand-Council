# One-time source publishing for the Grand Council repository.
# Run in PowerShell from the extracted source folder. Requires Git for Windows
# and authorization to push to the target repository. Does not overwrite history.
$ErrorActionPreference = 'Stop'
$Repo = 'https://github.com/thecakeisalie05/Grand-Council.git'
$Source = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw 'Git for Windows is required. Install Git from https://git-scm.com/download/win and rerun.'
}
$Temp = Join-Path $env:TEMP ('grand-council-publish-' + [guid]::NewGuid().ToString('N'))
try {
  git clone $Repo $Temp
  if ($LASTEXITCODE -ne 0) { throw 'Could not clone repository. Check your GitHub access.' }
  Get-ChildItem -LiteralPath $Source -Force | Where-Object {
    $_.Name -notin @('.git', '.venv', '.pytest_cache', '__pycache__', 'build', 'dist')
  } | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $Temp -Recurse -Force
  }
  Push-Location $Temp
  try {
    git add -A
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { Write-Host 'Repository already matches the current source.'; return }
    git commit -m 'Port Grand Council to Windows desktop with auto GPU inference'
    if ($LASTEXITCODE -ne 0) { throw 'Git commit failed; configure your Git name/email.' }
    git push origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw 'Git push failed. Authenticate with GitHub or check branch protections.' }
    Write-Host 'Source pushed. Windows desktop build will appear under GitHub Actions.' -ForegroundColor Green
  } finally { Pop-Location }
} finally {
  if (Test-Path $Temp) { Remove-Item -LiteralPath $Temp -Recurse -Force }
}
