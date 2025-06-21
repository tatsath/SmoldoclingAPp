#!/usr/bin/env python3
"""
Test PyMuPDF for reliable text extraction
"""

import os
import fitz  # PyMuPDF
import base64

def test_pymupdf_extraction():
    """Test PyMuPDF text extraction"""
    
    if not os.path.exists('sample-tables.pdf'):
        print("❌ sample-tables.pdf not found")
        return
    
    print("🧪 Testing PyMuPDF text extraction...")
    
    # Open PDF with PyMuPDF
    doc = fitz.open('sample-tables.pdf')
    
    print(f"📄 Document: {len(doc)} pages")
    
    texts = []
    images = []
    
    # Extract text from each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Extract text from page
        text_content = page.get_text()
        
        if text_content and text_content.strip():
            texts.append({
                'content': text_content.strip(),
                'page_number': page_num + 1
            })
            print(f"✅ Page {page_num + 1}: {len(text_content)} characters")
        else:
            print(f"⚠️ No text found on page {page_num + 1}")
        
        # Count images on page
        image_list = page.get_images()
        print(f"📷 Page {page_num + 1}: {len(image_list)} images")
    
    doc.close()
    
    print(f"\n📊 Results:")
    print(f"  Text chunks extracted: {len(texts)}")
    
    if texts:
        print(f"  First chunk preview: {texts[0]['content'][:100]}...")
        return True
    else:
        print("❌ No text extracted!")
        return False

if __name__ == "__main__":
    success = test_pymupdf_extraction()
    if success:
        print("\n🎉 PyMuPDF test PASSED!")
    else:
        print("\n💥 PyMuPDF test FAILED!") 