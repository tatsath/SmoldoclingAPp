#!/usr/bin/env python3
"""
Simple PDF Document Processor
Uses PDF Plumber for extraction and AWS Bedrock for analysis
"""

import os
import time
import json
import base64
import io
from typing import List, Dict, Any
from PIL import Image
import pandas as pd

# PDF processing
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Vector database
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# AWS Bedrock
try:
    import boto3
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

class SimpleDocumentProcessor:
    def __init__(self, 
                 aws_region: str = "ap-south-1",
                 model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        """Initialize simple document processor with AWS Bedrock"""
        self.aws_region = aws_region
        self.model_id = model_id
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize AWS Bedrock and ChromaDB"""
        # Initialize ChromaDB
        if CHROMA_AVAILABLE:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection("pdf_documents")
            print("✅ ChromaDB initialized")
        else:
            self.collection = None
            print("❌ ChromaDB not available")
        
        # Initialize AWS Bedrock
        if BEDROCK_AVAILABLE:
            try:
                self.bedrock_client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.aws_region,
                    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                    aws_session_token=os.environ.get('AWS_SESSION_TOKEN')
                )
                print(f"✅ AWS Bedrock initialized with model: {self.model_id}")
            except Exception as e:
                print(f"❌ Failed to initialize AWS Bedrock: {e}")
                raise ValueError(f"AWS Bedrock initialization failed: {e}")
        else:
            raise ValueError("boto3 not available for AWS Bedrock")
        
        # Check PDF Plumber
        if PDFPLUMBER_AVAILABLE:
            print("✅ PDF Plumber available")
        else:
            print("❌ PDF Plumber not available")
    
    def analyze_with_bedrock(self, prompt: str, image_base64: str = None) -> str:
        """Analyze content with AWS Bedrock"""
        try:
            if image_base64:
                # Vision model
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_base64
                                    }
                                }
                            ]
                        }
                    ]
                }
            else:
                # Text-only model
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        }
                    ]
                }
            
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json"
            )
            
            result = json.loads(response['body'].read())
            return result['content'][0]['text']
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def process_pdf_page(self, page, page_num: int) -> Dict[str, Any]:
        """Process a single PDF page"""
        try:
            # Extract text
            text_content = page.extract_text() or ""
            text_word_count = len(text_content.split())
            
            # Extract tables
            tables = page.extract_tables()
            table_count = len(tables)
            table_summaries = []
            
            # Analyze tables
            for i, table in enumerate(tables):
                if table:
                    table_df = pd.DataFrame(table)
                    table_text = table_df.to_string()
                    
                    prompt = f"""Analyze this table and provide a brief summary of its content and key data points:

Table Data:
{table_text}

Please provide a concise summary (2-3 sentences) of what this table contains."""
                    
                    summary = self.analyze_with_bedrock(prompt)
                    table_summaries.append(summary)
            
            # Convert page to image for analysis
            page_image = page.to_image()
            image = page_image.original
            
            # Resize if too large
            if max(image.size) > 2048:
                ratio = 2048 / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG', optimize=True)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Analyze image content
            image_prompt = f"""Analyze this document page image and provide:
1. A brief description of what you see
2. Any charts, graphs, or visual elements
3. Key information or data points visible

Text content from page: {text_content[:500]}

Please provide a concise analysis (3-4 sentences)."""
            
            image_analysis = self.analyze_with_bedrock(image_prompt, img_base64)
            
            # Overall page summary
            overall_prompt = f"""Provide a comprehensive summary of this document page:

Text Content: {text_content[:1000]}
Number of Tables: {table_count}
Table Summaries: {'; '.join(table_summaries) if table_summaries else 'None'}
Image Analysis: {image_analysis}

Please provide a detailed summary (4-5 sentences) covering the main content, key information, and any important data points."""
            
            overall_summary = self.analyze_with_bedrock(overall_prompt)
            
            return {
                "page_number": page_num,
                "text_word_count": text_word_count,
                "table_count": table_count,
                "table_summaries": table_summaries,
                "image_analysis": image_analysis,
                "overall_summary": overall_summary,
                "text_content": text_content,
                "processing_time": time.time()
            }
            
        except Exception as e:
            return {
                "page_number": page_num,
                "error": str(e),
                "text_word_count": 0,
                "table_count": 0,
                "table_summaries": [],
                "image_analysis": "Failed to analyze",
                "overall_summary": "Failed to analyze",
                "text_content": "",
                "processing_time": time.time()
            }
    
    def process_document(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process entire document with PDF Plumber"""
        print(f"📄 Processing document: {pdf_path}")
        start_time = time.time()
        
        if not PDFPLUMBER_AVAILABLE:
            raise ValueError("PDF Plumber is required but not available")
        
        results = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"📊 Found {total_pages} pages")
                
                for page_num in range(total_pages):
                    page = pdf.pages[page_num]
                    print(f"🔄 Processing page {page_num + 1}/{total_pages}")
                    
                    # Process page
                    result = self.process_pdf_page(page, page_num)
                    results.append(result)
                    
                    # Store in ChromaDB
                    if self.collection:
                        search_text = f"""
                        Page {result['page_number'] + 1}: {result['overall_summary']}
                        Text words: {result['text_word_count']}
                        Tables: {result['table_count']}
                        Table summaries: {'; '.join(result['table_summaries'])}
                        Image analysis: {result['image_analysis']}
                        """
                        
                        self.collection.add(
                            documents=[search_text],
                            metadatas=[{
                                "page_number": result["page_number"],
                                "text_word_count": result["text_word_count"],
                                "table_count": result["table_count"],
                                "filename": os.path.basename(pdf_path)
                            }],
                            ids=[f"{os.path.basename(pdf_path)}_page_{result['page_number']}"]
                        )
            
            total_time = time.time() - start_time
            print(f"✅ Document processing completed in {total_time:.2f}s")
            print(f"📈 Average time per page: {total_time/len(results):.2f}s")
            
            return results
            
        except Exception as e:
            print(f"❌ Document processing failed: {e}")
            raise
    
    def search_documents(self, query: str, n_results: int = 5) -> Dict:
        """Search documents in ChromaDB"""
        if not self.collection:
            return {}
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return {}
    
    def get_answer(self, question: str, document_content: str) -> str:
        """Generate a direct answer to a question using Bedrock"""
        prompt = f"""Based on the following document content, provide a direct and specific answer to this question: "{question}"

Document Content:
{document_content}

Please provide:
1. A direct answer to the question
2. Specific data points or facts that support your answer
3. The page numbers where this information was found

Answer:"""
        
        return self.analyze_with_bedrock(prompt) 