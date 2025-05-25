# FastAPI React Starter Setup Script for Windows
# This script sets up the development environment for the FastAPI React Starter project

# Stop on any error
$ErrorActionPreference = "Stop"

Write-Host "=== FastAPI React Starter Setup Script for Windows ===" -ForegroundColor Green

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Host "Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "Git is not installed. Please install Git and try again." -ForegroundColor Red
    exit 1
}

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python is not installed. Please install Python 3.8+ and try again." -ForegroundColor Red
    exit 1
}

# Check if Node.js is installed
try {
    $nodeVersion = node --version
    Write-Host "Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "Node.js is not installed. Please install Node.js 16+ and try again." -ForegroundColor Red
    exit 1
}

# Check if pnpm is installed
try {
    $pnpmVersion = pnpm --version
    Write-Host "pnpm found: $pnpmVersion" -ForegroundColor Green
} catch {
    Write-Host "pnpm is not installed. Installing pnpm..." -ForegroundColor Yellow
    npm install -g pnpm
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install pnpm. Please install it manually and try again." -ForegroundColor Red
        exit 1
    }
    Write-Host "pnpm installed successfully" -ForegroundColor Green
}

# Setup Git hooks
Write-Host "`nSetting up Git hooks for code formatting..." -ForegroundColor Cyan

# Install pre-commit
pip install pre-commit
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install pre-commit. Please check your Python installation." -ForegroundColor Red
    exit 1
}

# Find the pre-commit executable
$preCommitPath = $null

# Try to find pre-commit in the PATH
$preCommitCommand = Get-Command pre-commit -ErrorAction SilentlyContinue
if ($preCommitCommand) {
    $preCommitPath = $preCommitCommand.Source
    Write-Host "Found pre-commit in PATH: $preCommitPath" -ForegroundColor Green
} else {
    # Common locations for Python scripts in user directories
    $potentialLocations = @(
        "$env:APPDATA\Python\Scripts\pre-commit.exe",
        "$env:APPDATA\Python\Python*\Scripts\pre-commit.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python*\Scripts\pre-commit.exe",
        "$env:USERPROFILE\.local\bin\pre-commit"
    )

    foreach ($location in $potentialLocations) {
        $resolvedPaths = Resolve-Path -Path $location -ErrorAction SilentlyContinue
        if ($resolvedPaths) {
            foreach ($path in $resolvedPaths) {
                if (Test-Path $path) {
                    $preCommitPath = $path
                    Write-Host "Found pre-commit at: $preCommitPath" -ForegroundColor Green
                    break
                }
            }
        }
        if ($preCommitPath) { break }
    }

    # If still not found, try to get it from pip show
    if (-not $preCommitPath) {
        $pipShowOutput = pip show pre-commit
        $installLocation = $pipShowOutput | Where-Object { $_ -like "Location:*" }
        if ($installLocation) {
            $location = ($installLocation -split "Location: ")[1].Trim()
            $potentialPath = Join-Path -Path (Join-Path -Path $location -ChildPath "..") -ChildPath "Scripts\pre-commit.exe"
            if (Test-Path $potentialPath) {
                $preCommitPath = $potentialPath
                Write-Host "Found pre-commit at: $preCommitPath" -ForegroundColor Green
            }
        }
    }

    # If still not found
    if (-not $preCommitPath) {
        Write-Host "pre-commit.exe was installed but cannot be found." -ForegroundColor Red
        Write-Host "Please try running the following command in a new PowerShell window:" -ForegroundColor Yellow
        Write-Host "pip install pre-commit && pre-commit install" -ForegroundColor Yellow
        exit 1
    }
}

# Create .pre-commit-config.yaml if it doesn't exist
if (-not (Test-Path ".pre-commit-config.yaml")) {
    Write-Host "Creating .pre-commit-config.yaml..." -ForegroundColor Yellow
    @"
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
    -   id: trailing-whitespace
        stages: [pre-commit]
    -   id: end-of-file-fixer
        stages: [pre-commit]
    -   id: check-yaml
        stages: [pre-commit]
    -   id: check-added-large-files
        stages: [pre-commit]

-   repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
    -   id: black
        stages: [pre-commit]

-   repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.1.0
    hooks:
    -   id: prettier
        types_or: [javascript, jsx, ts, tsx, json, css, scss, markdown]
        stages: [pre-commit]
        files: ^frontend/src/.*\.(js|jsx|ts|tsx|json|css|scss|md)$
"@ | Out-File -FilePath ".pre-commit-config.yaml" -Encoding utf8
}

# Install the git hooks
& $preCommitPath install
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install pre-commit hooks. Please check your installation." -ForegroundColor Red
    exit 1
}

# Setup Backend
Write-Host "`nSetting up backend environment..." -ForegroundColor Cyan

# Create virtual environment if it doesn't exist
if (-not (Test-Path "backend\venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv backend\venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment. Please check your Python installation." -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment and install dependencies
Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
& backend\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to activate virtual environment. Please check your Python installation." -ForegroundColor Red
    exit 1
}

pip install -r backend\requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install backend dependencies. Please check your Python installation." -ForegroundColor Red
    exit 1
}

# Deactivate virtual environment
deactivate

# Setup Frontend
Write-Host "`nSetting up frontend environment..." -ForegroundColor Cyan
Set-Location frontend
Write-Host "Installing frontend dependencies with pnpm..." -ForegroundColor Yellow
pnpm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install frontend dependencies. Please check your pnpm installation." -ForegroundColor Red
    Set-Location ..
    exit 1
}
Set-Location ..

# Setup environment variables
Write-Host "`nSetting up environment variables..." -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from .env.example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
}

Write-Host "`nSetup completed successfully!" -ForegroundColor Green
Write-Host "`nTo start the backend, run:" -ForegroundColor Cyan
Write-Host "cd backend && ..\venv\Scripts\Activate.ps1 && uvicorn app.main:app --reload" -ForegroundColor Yellow
Write-Host "`nTo start the frontend, run:" -ForegroundColor Cyan
Write-Host "cd frontend && pnpm dev" -ForegroundColor Yellow
