#!/usr/bin/env bash
echo "======================================================================"
echo "          FLORAVISION AI - FLOWER SPECIES CLASSIFIER           "
echo "======================================================================"
echo ""
echo "[1/2] Checking Python environment..."
python3 --version || { echo "Python 3 is required"; exit 1; }

echo ""
echo "[2/2] Launching FloraVision AI Web Server..."
echo "  URL: http://localhost:5000"
echo ""
python3 app.py
