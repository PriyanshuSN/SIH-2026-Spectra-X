Write-Host 'Starting SpectraX FastAPI Backend on http://localhost:8000'
$env:PYTHONPATH = $PSScriptRoot
& '.\venv\Scripts\python.exe' -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
