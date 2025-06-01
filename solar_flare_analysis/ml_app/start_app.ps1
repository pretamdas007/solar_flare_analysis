# Solar Flare Analysis ML App Startup Script (PowerShell)
# This script starts both the Python ML backend and Next.js frontend

Write-Host "🌟 Starting Solar Flare Analysis ML Application" -ForegroundColor Yellow

# Function to check if port is in use
function Test-Port {
    param([int]$Port)
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
        return $connection.TcpTestSucceeded
    } catch {
        return $false
    }
}

# Function to start Python backend
function Start-PythonBackend {
    Write-Host "🐍 Starting Python ML Backend..." -ForegroundColor Green
    
    # Check if virtual environment is activated
    if (-not $env:VIRTUAL_ENV) {
        Write-Host "⚠️  Warning: No virtual environment detected. Trying to activate goesflareenv..." -ForegroundColor Yellow
        $venvPath = "..\goesflareenv\Scripts\Activate.ps1"
        if (Test-Path $venvPath) {
            & $venvPath
        } else {
            Write-Host "❌ Virtual environment not found. Please activate your Python environment first." -ForegroundColor Red
            exit 1
        }
    }
    
    # Install Python dependencies if needed
    try {
        python -c "import flask" 2>$null
    } catch {
        Write-Host "📦 Installing Python dependencies..." -ForegroundColor Cyan
        pip install flask flask-cors
    }
    
    # Start Python backend
    if (Test-Port -Port 5000) {
        Write-Host "⚠️  Port 5000 is already in use. Stopping existing process..." -ForegroundColor Yellow
        Get-Process | Where-Object {$_.ProcessName -eq "python" -and $_.CommandLine -like "*python_bridge.py*"} | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
    
    Write-Host "🚀 Launching Python ML backend on port 5000..." -ForegroundColor Green
    $pythonJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        python python_bridge.py --host 0.0.0.0 --port 5000
    }
    
    # Wait for backend to start
    Write-Host "⏳ Waiting for Python backend to initialize..." -ForegroundColor Cyan
    for ($i = 1; $i -le 30; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 1
            if ($response.StatusCode -eq 200) {
                Write-Host "✅ Python backend is ready!" -ForegroundColor Green
                return $pythonJob
            }
        } catch {
            # Continue waiting
        }
        Start-Sleep -Seconds 1
        if ($i -eq 30) {
            Write-Host "❌ Python backend failed to start" -ForegroundColor Red
            exit 1
        }
    }
    return $pythonJob
}

# Function to start Next.js frontend
function Start-NextjsFrontend {
    Write-Host "⚛️  Starting Next.js Frontend..." -ForegroundColor Green
    
    # Install Node.js dependencies if needed
    if (-not (Test-Path "node_modules")) {
        Write-Host "📦 Installing Node.js dependencies..." -ForegroundColor Cyan
        npm install
    }
    
    # Start Next.js development server
    if (Test-Port -Port 3000) {
        Write-Host "⚠️  Port 3000 is already in use." -ForegroundColor Yellow
        $continue = Read-Host "Continue anyway? (y/N)"
        if ($continue -ne "y" -and $continue -ne "Y") {
            exit 1
        }
    }
    
    Write-Host "🚀 Launching Next.js frontend on port 3000..." -ForegroundColor Green
    $nextjsJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        npm run dev
    }
    return $nextjsJob
}

# Main execution
Write-Host "🔧 Checking system requirements..." -ForegroundColor Cyan

# Check if Node.js is installed
try {
    node --version | Out-Null
} catch {
    Write-Host "❌ Node.js is not installed. Please install Node.js first." -ForegroundColor Red
    exit 1
}

# Check if Python is installed
try {
    python --version | Out-Null
} catch {
    Write-Host "❌ Python is not installed. Please install Python first." -ForegroundColor Red
    exit 1
}

Write-Host "✅ System requirements satisfied" -ForegroundColor Green

# Start services
$pythonJob = Start-PythonBackend
$nextjsJob = Start-NextjsFrontend

Write-Host ""
Write-Host "🎉 Solar Flare Analysis ML Application is running!" -ForegroundColor Yellow
Write-Host ""
Write-Host "📊 Frontend:  http://localhost:3000" -ForegroundColor Cyan
Write-Host "🔧 Backend:   http://localhost:5000" -ForegroundColor Cyan
Write-Host "❤️  Health:   http://localhost:5000/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

# Wait for user input to stop
try {
    while ($true) {
        Start-Sleep -Seconds 1
        # Check if jobs are still running
        if ($pythonJob.State -eq "Failed" -or $nextjsJob.State -eq "Failed") {
            Write-Host "❌ One or more services failed" -ForegroundColor Red
            break
        }
    }
} finally {
    Write-Host "🛑 Shutting down services..." -ForegroundColor Yellow
    if ($pythonJob) {
        Stop-Job -Job $pythonJob -PassThru | Remove-Job
        Write-Host "✅ Python backend stopped" -ForegroundColor Green
    }
    if ($nextjsJob) {
        Stop-Job -Job $nextjsJob -PassThru | Remove-Job
        Write-Host "✅ Next.js frontend stopped" -ForegroundColor Green
    }
    Write-Host "👋 Goodbye!" -ForegroundColor Yellow
}