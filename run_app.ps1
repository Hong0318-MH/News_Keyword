$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $ProjectRoot "vendor"
Set-Location $ProjectRoot
python -m streamlit.web.cli run app.py --global.developmentMode false --server.port 8501 --server.headless true
