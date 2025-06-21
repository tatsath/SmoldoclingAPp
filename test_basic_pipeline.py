#!/usr/bin/env python3
"""
Test basic pipeline without any model loading
"""

import os
from docling.document_converter import DocumentConverter
from docling_core.types.doc import TextItem, SectionHeaderItem

def test_basic_pipeline():
    """Test basic DocumentConverter without custom options"""
    
    if not os.path.exists('sample-tables.pdf'):
        print("❌ sample-tables.pdf not found")
        return
    
    print("🧪 Testing basic pipeline without model loading...")
    
    # Use basic DocumentConverter (no custom options)
    converter = DocumentConverter()
    
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
    
    # Method 2: Fallback to markdown export
    if not texts:
        print("🔄 Trying markdown export...")
        try:
            md_content = doc.export_to_markdown()
            if md_content and md_content.strip():
                sections = md_content.split('\n## ')
                for i, section in enumerate(sections):
                    if section.strip():
                        texts.append({
                            'content': section.strip(),
                            'page_number': i + 1
                        })
                print(f"✅ Markdown export: {len(texts)} sections")
        except Exception as e:
            print(f"❌ Markdown export failed: {e}")
    
    print(f"\n📊 Results:")
    print(f"  Text chunks extracted: {len(texts)}")
    print(f"  Pages with text: {len(page_texts)}")
    
    if texts:
        print(f"  First chunk preview: {texts[0]['content'][:100]}...")
        return True
    else:
        print("❌ No text extracted!")
        return False

if __name__ == "__main__":
    success = test_basic_pipeline()
    if success:
        print("\n🎉 Basic pipeline test PASSED!")
    else:
        print("\n💥 Basic pipeline test FAILED!") 