#!/bin/bash

# Script to run the docling app with correct environment

echo "🚀 Starting ECC Memo Generator with docling..."

# Activate the docling environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate docling

# Set environment variables to avoid meta tensor errors
export DOCLING_DISABLE_MODELS=1
export TORCH_DEVICE=cpu
export DOCLING_USE_CPU=1

# Run the Streamlit app
echo "📱 Starting Streamlit app on http://localhost:8501"
streamlit run app_docling.py --server.port 8501 