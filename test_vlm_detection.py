#!/usr/bin/env python3
"""
Test VLM detection
"""

import os

# Set environment variables to avoid meta tensor errors
os.environ['DOCLING_DISABLE_MODELS'] = '1'
os.environ['TORCH_DEVICE'] = 'cpu'
os.environ['DOCLING_USE_CPU'] = '1'

print("🔍 Testing VLM detection...")

# Test basic imports
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline
    print("✅ Basic docling imports successful")
except ImportError as e:
    print(f"❌ Basic import error: {e}")
    exit(1)

# Test VLM imports
try:
    from docling.datamodel.pipeline_options import (
        smoldocling_vlm_conversion_options,
        smoldocling_vlm_mlx_conversion_options
    )
    print("✅ VLM model specs import successful")
    print(f"Default VLM options: {smoldocling_vlm_conversion_options}")
    print(f"MLX VLM options: {smoldocling_vlm_mlx_conversion_options}")
        
    VLM_AVAILABLE = True
except ImportError as e:
    print(f"❌ VLM import error: {e}")
    VLM_AVAILABLE = False

print(f"\n📋 VLM_AVAILABLE: {VLM_AVAILABLE}")

if VLM_AVAILABLE:
    print("🎉 VLM options should appear in the UI!")
else:
    print("⚠️ Only Basic Pipeline will be available in the UI") 