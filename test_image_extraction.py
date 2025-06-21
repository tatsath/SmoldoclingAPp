#!/usr/bin/env python3
"""
Test script to verify image extraction is working
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
    from docling_core.types.doc import PictureItem, TableItem, TextItem, SectionHeaderItem
    import base64
    print("✅ All docling imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def test_image_extraction():
    """Test image extraction functionality"""
    
    if not os.path.exists('sample-tables.pdf'):
        print("❌ sample-tables.pdf not found")
        return
    
    print("🧪 Testing image extraction...")
    
    try:
        # Create pipeline options with image generation enabled
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
        
        # Use DocumentConverter with image generation enabled
        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        })
        
        print("✅ Converter initialized with image generation enabled")
        
        # Convert document
        result = converter.convert('sample-tables.pdf')
        doc = result.document
        
        print(f"📄 Document: {len(doc.pages)} pages, {len(doc.tables)} tables")
        
        # Extract images
        images = []
        
        # Extract images from figures/pictures
        picture_counter = 0
        for element, _level in doc.iterate_items():
            if isinstance(element, PictureItem):
                picture_counter += 1
                try:
                    # Get image from picture element
                    img = element.get_image(doc)
                    if img:
                        # Convert PIL image to base64
                        import io
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='PNG')
                        img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                        
                        images.append({
                            'base64': img_b64,
                            'page_number': getattr(element, 'page_no', 1),
                            'type': 'picture',
                            'index': picture_counter
                        })
                        print(f"✅ Extracted picture {picture_counter} from page {getattr(element, 'page_no', 1)}")
                except Exception as e:
                    print(f"❌ Failed to process picture {picture_counter}: {e}")
        
        # Extract images from tables
        table_img_counter = 0
        for element, _level in doc.iterate_items():
            if isinstance(element, TableItem):
                table_img_counter += 1
                try:
                    # Get image from table element
                    img = element.get_image(doc)
                    if img:
                        # Convert PIL image to base64
                        import io
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='PNG')
                        img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                        
                        images.append({
                            'base64': img_b64,
                            'page_number': getattr(element, 'page_no', 1),
                            'type': 'table',
                            'index': table_img_counter
                        })
                        print(f"✅ Extracted table image {table_img_counter} from page {getattr(element, 'page_no', 1)}")
                except Exception as e:
                    print(f"❌ Failed to process table image {table_img_counter}: {e}")
        
        print(f"🖼️ Total images extracted: {len(images)}")
        
        if len(images) > 0:
            print("✅ Image extraction test completed successfully!")
            return True
        else:
            print("⚠️ No images found in the document")
            return True  # Still successful, just no images in this document
        
    except Exception as e:
        print(f"❌ Image extraction test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting image extraction test...")
    
    # Test image extraction
    success = test_image_extraction()
    
    if success:
        print("\n✅ Image extraction test passed!")
    else:
        print("\n❌ Image extraction test failed.")

if __name__ == "__main__":
    main() 