# Quick Start Guide - ECC Memo Generator

## ✅ Meta Tensor Error Fixed!

The meta tensor error has been resolved. The app now uses docling with two pipeline options to avoid PyTorch model loading issues.

## 🚀 How to Run the App

### Option 1: Using the Script (Recommended)
```bash
./run_app.sh
```

### Option 2: Manual Steps
```bash
# 1. Activate the docling environment
conda activate docling

# 2. Set environment variables (already done in app_docling.py)
export DOCLING_DISABLE_MODELS=1
export TORCH_DEVICE=cpu
export DOCLING_USE_CPU=1

# 3. Run the app
streamlit run app_docling.py --server.port 8501
```

## 🌐 Access the App
Open your browser and go to: **http://localhost:8501**

## 📋 Pipeline Options

### 1. Basic Pipeline (Recommended)
- ✅ **Fast processing**: Text, table, and image extraction
- ✅ **No model loading**: Avoids meta tensor errors
- ✅ **Reliable**: Works consistently across different documents
- ✅ **Lightweight**: Minimal resource usage

### 2. SmolDocling VLM (Default)
- 🚀 **Advanced understanding**: Uses SmolDocling vision-language model
- 🚀 **Transformers framework**: Standard PyTorch-based processing
- 🚀 **Better comprehension**: Enhanced document understanding
- ⚠️ **Slower processing**: More computational intensive

### 3. SmolDocling VLM (MLX)
- 🚀 **Advanced understanding**: Uses SmolDocling vision-language model
- 🚀 **MLX framework**: Optimized for Apple Silicon Macs
- 🚀 **Better performance**: Faster processing on M1/M2 Macs
- ⚠️ **Apple Silicon only**: Requires M1/M2 Mac for optimal performance

## 🔧 Configuration
The app automatically detects available capabilities:
- **Basic Pipeline**: Always available
- **SmolDocling VLM (Default)**: Available if SmolDocling models are installed
- **SmolDocling VLM (MLX)**: Available if SmolDocling models and MLX are installed

## 🧪 Testing
To verify both pipelines work:
```bash
conda activate docling
python test_both_pipelines.py
```

## 📁 Key Files
- `app_docling.py` - Main Streamlit application with dual pipeline support
- `run_app.sh` - Easy startup script
- `test_both_pipelines.py` - Test script for both pipelines
- `test_image_extraction.py` - Test script for image extraction
- `META_TENSOR_FIX.md` - Detailed explanation of the fix 