#!/bin/bash
set -e

echo "========================================"
echo "  PrintFarmEnv Pre-Submission Validator "
echo "========================================"

echo "[1/4] Checking openenv-core schema validation..."
if command -v openenv &> /dev/null; then
    openenv validate locally
    echo "✅ OpenEnv schema passed validation!"
else
    echo "⚠️ 'openenv' CLI tool not found. Trying python validation..."
    # A simple python load test to verify syntax/models import correctly
    python -c "import printfarm_env.models; print('✅ Syntax & Pydantic Checks OK')"
fi

echo ""
echo "[2/4] Testing Baseline Environment (inference.py)..."
# Set dummy token so it doesn't try actual openai fetch without keys
export HF_TOKEN="dummy_token_for_validation"
# Limit runtime test to just check syntax without stalling
if python inference.py > /dev/null 2>&1; then
   echo "✅ Target environment code executes without exceptions!"
else
   echo "❌ Baseline loop execution failed."
   exit 1
fi

echo ""
echo "[3/4] Trying to build Docker image locally..."
if docker info > /dev/null 2>&1; then
    docker build -t printfarm .
    echo "✅ Docker build succeeded!"
else
    echo "⚠️ Docker daemon is not running (or not accessible). Skipping Docker build test."
fi

echo ""
echo "[4/4] Verification check checklist complete!"
echo "----------------------------------------"
echo "Before you submit final code to Hugging Face, remember to:"
echo "1. Verify 'Ping Test -> curl -I https://your-space.hf.space' returns 200 OK after deployment."
echo "2. Confirm your HF space interface boots gracefully!"
echo "========================================"
echo "Great work, you are ready to ship! 🚀"
