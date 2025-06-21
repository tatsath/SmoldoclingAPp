# Quick Start Guide - ECC Memo Generator

## ✅ Meta Tensor Error Fixed!

The meta tensor error has been resolved. The app now uses docling with minimal configuration to avoid PyTorch model loading issues.

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

## 📋 What Works Now
- ✅ Document upload (PDF, DOC, DOCX, PPT, PPTX, TXT)
- ✅ Text extraction using docling
- ✅ Table detection and extraction
- ✅ ChromaDB vector storage
- ✅ AWS Bedrock integration
- ✅ Memo generation with semantic search
- ✅ **No more meta tensor errors!**

## 🔧 Configuration
The app uses minimal docling configuration:
- Text extraction: ✅ Enabled (PDF backend)
- Table detection: ✅ Enabled
- OCR: ❌ Disabled (to avoid model loading)
- Image processing: ❌ Disabled (to avoid model loading)
- VLM processing: ❌ Disabled (to avoid model loading)

## 🧪 Testing
To verify the fix works:
```bash
conda activate docling
python test_docling_fixed.py
```

## 📁 Key Files
- `app_docling.py` - Main Streamlit application
- `run_app.sh` - Easy startup script
- `test_docling_fixed.py` - Test script to verify the fix
- `META_TENSOR_FIX.md` - Detailed explanation of the fix 