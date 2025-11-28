# Quick Start Script - Tanui Assistant
# This script kills old processes and starts fresh servers

Write-Host "🔧 Killing old Node.js processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*node*"} | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "🧹 Cleaning Next.js cache..." -ForegroundColor Yellow
Remove-Item -Recurse -Force "src\web\client\.next" -ErrorAction SilentlyContinue
Remove-Item -Force "src\web\client\.next\dev\lock" -ErrorAction SilentlyContinue

Write-Host "✅ Starting servers..." -ForegroundColor Green
Write-Host ""
Write-Host "📌 Backend will run on: http://localhost:5000" -ForegroundColor Cyan
Write-Host "📌 Frontend will run on: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""

# Start backend in new terminal
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "python src\web\app.py" -WorkingDirectory $PWD

# Wait a moment
Start-Sleep -Seconds 2

# Start frontend in new terminal
Set-Location "src\web\client"
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "npm run dev" -WorkingDirectory $PWD

Write-Host "✅ Servers started!

" -ForegroundColor Green
Write-Host "Visit: " -NoNewline; Write-Host "http://localhost:3000" -ForegroundColor Cyan
