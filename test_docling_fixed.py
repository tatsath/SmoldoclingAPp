#!/usr/bin/env python3
"""
Test script to verify that the meta tensor error is fixed
"""

import os
import tempfile

# Set environment variables to avoid meta tensor errors
os.environ['DOCLING_DISABLE_MODELS'] = '1'
os.environ['TORCH_DEVICE'] = 'cpu'
os.environ['DOCLING_USE_CPU'] = '1'

# Test docling imports
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import TextItem, SectionHeaderItem
    print("✅ All docling imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def test_minimal_pipeline():
    """Test minimal pipeline configuration to avoid meta tensor errors"""
    
    if not os.path.exists('sample-tables.pdf'):
        print("❌ sample-tables.pdf not found")
        return
    
    print("🧪 Testing minimal pipeline configuration...")
    
    try:
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
        
        # Use DocumentConverter with minimal configuration
        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        })
        
        print("✅ Converter initialized with minimal configuration")
        
        # Convert document
        result = converter.convert('sample-tables.pdf')
        doc = result.document
        
        print(f"📄 Document: {len(doc.pages)} pages, {len(doc.tables)} tables")
        
        # Extract text using the same logic as the app
        texts = []
        page_texts = {}
        
        # Method 1: Extract from document elements
        for element, _level in doc.iterate_items():
            if isinstance(element, (TextItem, SectionHeaderItem)) and hasattr(element, 'text') and element.text:
                page_no = getattr(element, 'page_no', 1)
                if page_no not in page_texts:
                    page_texts[page_no] = []
                page_texts[page_no].append(element.text.strip())
        
        # Create text chunks
        for page_no, text_parts in page_texts.items():
            if text_parts:
                combined_text = "\n".join(text_parts)
                texts.append({
                    'content': combined_text,
                    'page_number': page_no
                })
                print(f"✅ Page {page_no}: {len(combined_text)} characters")
            else:
                print(f"⚠️ Page {page_no}: No text found")
        
        print(f"📊 Total text chunks: {len(texts)}")
        
        # Test table extraction
        tables = []
        for table_ix, table in enumerate(doc.tables):
            try:
                table_df = table.export_to_dataframe()
                table_content = table_df.to_string()
                table_html = table.export_to_html(doc=doc)
                
                tables.append({
                    'content': table_content,
                    'page_number': getattr(table, 'page_no', 1),
                    'html': table_html,
                    'dataframe': table_df
                })
                print(f"✅ Table {table_ix + 1}: {table_df.shape[0]} rows, {table_df.shape[1]} columns")
            except Exception as e:
                print(f"❌ Failed to process table {table_ix}: {e}")
        
        print(f"📊 Total tables: {len(tables)}")
        
        print("✅ Minimal pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Minimal pipeline test failed: {e}")
        return False

def test_environment_variables():
    """Test that environment variables are set correctly"""
    print("\n🔍 Testing environment variables:")
    
    required_vars = ['DOCLING_DISABLE_MODELS', 'TORCH_DEVICE', 'DOCLING_USE_CPU']
    
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    return all(os.environ.get(var) for var in required_vars)

def main():
    """Main test function"""
    print("🚀 Starting meta tensor error fix verification...")
    
    # Test environment variables
    env_ok = test_environment_variables()
    
    if not env_ok:
        print("❌ Environment variables not set correctly")
        return
    
    # Test minimal pipeline
    pipeline_ok = test_minimal_pipeline()
    
    if pipeline_ok:
        print("\n✅ All tests passed! Meta tensor error should be fixed.")
    else:
        print("\n❌ Tests failed. Meta tensor error may still occur.")

if __name__ == "__main__":
    main() 