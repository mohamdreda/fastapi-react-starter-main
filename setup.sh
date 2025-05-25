#!/bin/bash
# FastAPI React Starter Setup Script for Linux/Mac
# This script sets up the development environment for the FastAPI React Starter project

# Stop on any error
set -e

echo -e "\e[32m=== FastAPI React Starter Setup Script for Linux/Mac ===\e[0m"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "\e[31mGit is not installed. Please install Git and try again.\e[0m"
    exit 1
fi
echo -e "\e[32mGit found: $(git --version)\e[0m"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "\e[31mPython 3 is not installed. Please install Python 3.8+ and try again.\e[0m"
    exit 1
fi
echo -e "\e[32mPython found: $(python3 --version)\e[0m"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "\e[31mNode.js is not installed. Please install Node.js 16+ and try again.\e[0m"
    exit 1
fi
echo -e "\e[32mNode.js found: $(node --version)\e[0m"

# Check if pnpm is installed
if ! command -v pnpm &> /dev/null; then
    echo -e "\e[33mpnpm is not installed. Installing pnpm...\e[0m"
    npm install -g pnpm
    if [ $? -ne 0 ]; then
        echo -e "\e[31mFailed to install pnpm. Please install it manually and try again.\e[0m"
        exit 1
    fi
    echo -e "\e[32mpnpm installed successfully\e[0m"
else
    echo -e "\e[32mpnpm found: $(pnpm --version)\e[0m"
fi

# Setup Git hooks
echo -e "\n\e[36mSetting up Git hooks for code formatting...\e[0m"

# Install pre-commit
python3 -m pip install pre-commit
if [ $? -ne 0 ]; then
    echo -e "\e[31mFailed to install pre-commit. Please check your Python installation.\e[0m"
    exit 1
fi

# Find the pre-commit executable path
PRE_COMMIT_PATH=$(which pre-commit 2>/dev/null)
if [ -z "$PRE_COMMIT_PATH" ]; then
    # Try to find pre-commit in user's Python bin directory
    PYTHON_USER_BASE=$(python3 -m site --user-base)
    PRE_COMMIT_PATH="$PYTHON_USER_BASE/bin/pre-commit"
    if [ ! -x "$PRE_COMMIT_PATH" ]; then
        echo -e "\e[31mpre-commit executable not found in PATH or user's Python bin directory.\e[0m"
        echo -e "\e[33mYou may need to add Python's bin directory to your PATH manually.\e[0m"
        echo -e "\e[33mTry: export PATH=\"$PYTHON_USER_BASE/bin:\$PATH\"\e[0m"
        exit 1
    fi
fi

echo -e "\e[32mFound pre-commit at: $PRE_COMMIT_PATH\e[0m"

# Create .pre-commit-config.yaml if it doesn't exist
if [ ! -f ".pre-commit-config.yaml" ]; then
    echo -e "\e[33mCreating .pre-commit-config.yaml...\e[0m"
    cat > .pre-commit-config.yaml << EOF
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
EOF
fi

# Install the git hooks
"$PRE_COMMIT_PATH" install
if [ $? -ne 0 ]; then
    echo -e "\e[31mFailed to install pre-commit hooks. Please check your installation.\e[0m"
    exit 1
fi

# Setup Backend
echo -e "\n\e[36mSetting up backend environment...\e[0m"

# Create virtual environment if it doesn't exist
if [ ! -d "backend/venv" ]; then
    echo -e "\e[33mCreating Python virtual environment...\e[0m"
    python3 -m venv backend/venv
    if [ $? -ne 0 ]; then
        echo -e "\e[31mFailed to create virtual environment. Please check your Python installation.\e[0m"
        exit 1
    fi
fi

# Activate virtual environment and install dependencies
echo -e "\e[33mInstalling backend dependencies...\e[0m"
source backend/venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "\e[31mFailed to activate virtual environment. Please check your Python installation.\e[0m"
    exit 1
fi

pip install -r backend/requirements.txt
if [ $? -ne 0 ]; then
    echo -e "\e[31mFailed to install backend dependencies. Please check your Python installation.\e[0m"
    exit 1
fi

# Deactivate virtual environment
deactivate

# Setup Frontend
echo -e "\n\e[36mSetting up frontend environment...\e[0m"
cd frontend
echo -e "\e[33mInstalling frontend dependencies with pnpm...\e[0m"
pnpm install
if [ $? -ne 0 ]; then
    echo -e "\e[31mFailed to install frontend dependencies. Please check your pnpm installation.\e[0m"
    cd ..
    exit 1
fi
cd ..

# Setup environment variables
echo -e "\n\e[36mSetting up environment variables...\e[0m"
if [ ! -f ".env" ]; then
    echo -e "\e[33mCreating .env file from .env.example...\e[0m"
    cp .env.example .env
fi

echo -e "\n\e[32mSetup completed successfully!\e[0m"
echo -e "\n\e[36mTo start the backend, run:\e[0m"
echo -e "\e[33mcd backend && source venv/bin/activate && uvicorn app.main:app --reload\e[0m"
echo -e "\n\e[36mTo start the frontend, run:\e[0m"
echo -e "\e[33mcd frontend && pnpm dev\e[0m"
