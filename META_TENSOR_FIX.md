# Meta Tensor Error Fix

## Problem
The error "Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device" was occurring when using docling with PyTorch models.

## Root Cause
This error occurs when PyTorch models are loaded in meta state (to save memory) and then the code tries to move them to a device using `.to()` instead of `.to_empty()`.

## Solution
The fix involved several changes to avoid model loading and meta tensor issues:

### 1. Environment Variables
Added environment variables at the top of `app_docling.py` to disable model loading:

```python
# Set environment variables to avoid meta tensor errors
os.environ['DOCLING_DISABLE_MODELS'] = '1'
os.environ['TORCH_DEVICE'] = 'cpu'
os.environ['DOCLING_USE_CPU'] = '1'
```

### 2. Minimal Pipeline Configuration
Created a minimal pipeline configuration that disables all model-dependent features:

```python
# Create minimal pipeline options to avoid model loading
pipeline_options = PdfPipelineOptions()
pipeline_options.force_backend_text = True  # Use PDF backend for text extraction
pipeline_options.do_ocr = False  # Disable OCR to avoid model loading
pipeline_options.do_table_structure = True  # Enable table detection
pipeline_options.do_picture_classification = False  # Disable to avoid model loading
pipeline_options.do_picture_description = False  # Disable to avoid model loading
pipeline_options.do_code_enrichment = False  # Disable to avoid model loading
pipeline_options.do_formula_enrichment = False  # Disable to avoid model loading
pipeline_options.generate_page_images = False  # Disable image generation
pipeline_options.generate_picture_images = False  # Disable image generation
```

### 3. DocumentConverter Configuration
Used the minimal configuration with the DocumentConverter:

```python
# Use DocumentConverter with minimal configuration
default_converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
})
```

### 4. Disabled Image Extraction
Removed image extraction functionality to avoid model loading:

```python
# Extract images from document (disabled to avoid model loading)
# Note: Image extraction is disabled to avoid meta tensor errors
st.info("🖼️ Image extraction disabled to avoid model loading")
```

## Features Retained
- ✅ Text extraction from PDF backend
- ✅ Table structure detection and extraction
- ✅ Document element iteration
- ✅ Markdown export as fallback
- ✅ Support for multiple document formats (PDF, DOC, DOCX, PPT, PPTX, TXT)

## Features Disabled
- ❌ OCR (Optical Character Recognition)
- ❌ Image extraction and processing
- ❌ Picture classification
- ❌ Picture description
- ❌ Code enrichment
- ❌ Formula enrichment
- ❌ VLM (Vision Language Model) processing

## Testing
The fix was verified using `test_docling_fixed.py` which:
1. Tests environment variable configuration
2. Tests minimal pipeline initialization
3. Tests text and table extraction
4. Confirms no meta tensor errors occur

## Result
The app now works without meta tensor errors while still providing robust text and table extraction capabilities using docling's basic pipeline features. 