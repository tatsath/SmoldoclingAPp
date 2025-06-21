#!/usr/bin/env python3
"""
Test script to verify both basic and VLM pipelines work
"""

import os

# Set environment variables to avoid meta tensor errors
os.environ['DOCLING_DISABLE_MODELS'] = '1'
os.environ['TORCH_DEVICE'] = 'cpu'
os.environ['DOCLING_USE_CPU'] = '1'

# Test docling imports
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling_core.types.doc import PictureItem, TableItem, TextItem, SectionHeaderItem
    
    # Test VLM imports
    try:
        from docling.datamodel import vlm_model_specs
        VLM_AVAILABLE = True
        print("✅ All docling imports successful (VLM available)")
    except ImportError:
        VLM_AVAILABLE = False
        print("✅ Basic docling imports successful (VLM not available)")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def test_basic_pipeline():
    """Test basic pipeline functionality"""
    
    if not os.path.exists('sample-tables.pdf'):
        print("❌ sample-tables.pdf not found")
        return False
    
    print("\n🧪 Testing Basic Pipeline...")
    
    try:
        # Create basic pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.force_backend_text = True
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.do_picture_classification = False
        pipeline_options.do_picture_description = False
        pipeline_options.do_code_enrichment = False
        pipeline_options.do_formula_enrichment = False
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 1.5
        
        # Use DocumentConverter with basic configuration
        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        })
        
        print("✅ Basic converter initialized")
        
        # Convert document
        result = converter.convert('sample-tables.pdf')
        doc = result.document
        
        print(f"📄 Document: {len(doc.pages)} pages, {len(doc.tables)} tables")
        
        # Test text extraction
        texts = []
        page_texts = {}
        for element, _level in doc.iterate_items():
            if isinstance(element, (TextItem, SectionHeaderItem)) and hasattr(element, 'text') and element.text:
                page_no = getattr(element, 'page_no', 1)
                if page_no not in page_texts:
                    page_texts[page_no] = []
                page_texts[page_no].append(element.text.strip())
        
        for page_no, text_parts in page_texts.items():
            if text_parts:
                combined_text = "\n".join(text_parts)
                texts.append({
                    'content': combined_text,
                    'page_number': page_no
                })
        
        print(f"✅ Basic pipeline: {len(texts)} text chunks, {len(doc.tables)} tables")
        return True
        
    except Exception as e:
        print(f"❌ Basic pipeline test failed: {e}")
        return False

def test_vlm_pipeline():
    """Test VLM pipeline functionality"""
    
    if not VLM_AVAILABLE:
        print("\n⚠️ VLM not available, skipping VLM tests")
        return True
    
    if not os.path.exists('sample-tables.pdf'):
        print("❌ sample-tables.pdf not found")
        return False
    
    print("\n🧪 Testing VLM Pipelines...")
    
    # Test 1: SmolDocling VLM (Default)
    print("\n📋 Testing SmolDocling VLM (Default)...")
    try:
        # Use VLM pipeline with default settings (transformers)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                ),
            }
        )
        
        print("✅ Default VLM converter initialized")
        
        # Convert document
        result = converter.convert('sample-tables.pdf')
        doc = result.document
        
        print(f"📄 Document: {len(doc.pages)} pages, {len(doc.tables)} tables")
        
        # Test markdown export
        try:
            md_content = doc.export_to_markdown()
            print(f"✅ Default VLM: Markdown export successful ({len(md_content)} characters)")
        except Exception as e:
            print(f"⚠️ Default VLM: Markdown export failed: {e}")
        
    except Exception as e:
        print(f"❌ Default VLM test failed: {e}")
        return False
    
    # Test 2: SmolDocling VLM (MLX)
    print("\n📋 Testing SmolDocling VLM (MLX)...")
    try:
        # Use VLM pipeline with MLX options
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.SMOLDOCLING_MLX,
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )
        
        print("✅ MLX VLM converter initialized")
        
        # Convert document
        result = converter.convert('sample-tables.pdf')
        doc = result.document
        
        print(f"📄 Document: {len(doc.pages)} pages, {len(doc.tables)} tables")
        
        # Test markdown export
        try:
            md_content = doc.export_to_markdown()
            print(f"✅ MLX VLM: Markdown export successful ({len(md_content)} characters)")
        except Exception as e:
            print(f"⚠️ MLX VLM: Markdown export failed: {e}")
        
    except Exception as e:
        print(f"❌ MLX VLM test failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 Starting pipeline tests...")
    
    # Test basic pipeline
    basic_ok = test_basic_pipeline()
    
    # Test VLM pipelines
    vlm_ok = test_vlm_pipeline()
    
    if basic_ok and vlm_ok:
        print("\n✅ All pipeline tests passed!")
        print("📋 Summary:")
        print("  - Basic Pipeline: ✅ Working")
        if VLM_AVAILABLE:
            print("  - SmolDocling VLM (Default): ✅ Working")
            print("  - SmolDocling VLM (MLX): ✅ Working")
        else:
            print("  - SmolDocling VLM (Default): ⚠️ Not available")
            print("  - SmolDocling VLM (MLX): ⚠️ Not available")
    else:
        print("\n❌ Some pipeline tests failed.")

if __name__ == "__main__":
    main() 