#!/usr/bin/env python3
"""
Simple PDF Document Processor with Q&A
Uses PDF Plumber for extraction and AWS Bedrock for analysis
"""

import streamlit as st
import os
import tempfile
import time
import json
from PIL import Image
import pandas as pd
import pdfplumber
import base64
import io
import boto3
import chromadb
from chromadb.config import Settings
import pickle
from typing import Dict, List, Any
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Page configuration
st.set_page_config(
    page_title="Simple PDF Processor",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "processor" not in st.session_state:
    st.session_state.processor = None
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = None
if "collection" not in st.session_state:
    st.session_state.collection = None
if "multivector_data" not in st.session_state:
    st.session_state.multivector_data = {}
if "multivector_retriever" not in st.session_state:
    st.session_state.multivector_retriever = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "byte_store" not in st.session_state:
    st.session_state.byte_store = None

def initialize_bedrock(aws_region, model_id):
    """Initialize AWS Bedrock client"""
    try:
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=aws_region,
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.environ.get('AWS_SESSION_TOKEN')
        )
        return bedrock_client
    except Exception as e:
        st.error(f"Failed to initialize AWS Bedrock: {e}")
        return None

def initialize_chroma():
    """Initialize ChromaDB"""
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection("pdf_documents")
        return chroma_client, collection
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {e}")
        return None, None

def get_all_chroma_collections():
    """Get all available ChromaDB collections"""
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collections = chroma_client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        st.error(f"Failed to get ChromaDB collections: {e}")
        return []

def get_collection_by_name(collection_name):
    """Get a specific ChromaDB collection by name"""
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection(collection_name)
        return collection
    except Exception as e:
        st.error(f"Failed to get collection '{collection_name}': {e}")
        return None

def initialize_multivector_retriever():
    """Initialize MultiVectorRetriever with LangChain components"""
    try:
        # Initialize embeddings using HuggingFace (no API key needed)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize vectorstore
        vectorstore = Chroma(
            collection_name="multivector_documents",
            embedding_function=embeddings,
            persist_directory="./multivector_db"
        )
        
        # Initialize byte store for parent documents
        byte_store = InMemoryByteStore()
        id_key = "doc_id"
        
        # Initialize MultiVectorRetriever
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=byte_store,
            id_key=id_key
        )
        
        return retriever, vectorstore, byte_store
    except Exception as e:
        st.error(f"Failed to initialize MultiVectorRetriever: {e}")
        return None, None, None

def save_multivector_data(data: Dict[str, Any], filename: str):
    """Save multivector data to pickle file"""
    try:
        pickle_path = f"./multivector_data_{filename}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        return pickle_path
    except Exception as e:
        st.error(f"Failed to save multivector data: {e}")
        return None

def load_multivector_data(filename: str) -> Dict[str, Any]:
    """Load multivector data from pickle file"""
    try:
        pickle_path = f"./multivector_data_{filename}.pkl"
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        return {}
    except Exception as e:
        st.error(f"Failed to load multivector data: {e}")
        return {}

def save_embeddings_to_pickle(embeddings_data: Dict[str, Any], filename: str):
    """Save embeddings data to pickle file"""
    try:
        pickle_path = f"./embeddings_{filename}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(embeddings_data, f)
        return pickle_path
    except Exception as e:
        st.error(f"Failed to save embeddings: {e}")
        return None

def load_embeddings_from_pickle(filename: str) -> Dict[str, Any]:
    """Load embeddings data from pickle file"""
    try:
        pickle_path = f"./embeddings_{filename}.pkl"
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        return {}
    except Exception as e:
        st.error(f"Failed to load embeddings: {e}")
        return {}

def analyze_with_bedrock(bedrock_client, model_id, prompt, image_base64=None):
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
        
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']
        
    except Exception as e:
        return f"Analysis failed: {str(e)}"

def process_pdf_page(page, page_num, bedrock_client, model_id):
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
                
                summary = analyze_with_bedrock(bedrock_client, model_id, prompt)
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
        
        image_analysis = analyze_with_bedrock(bedrock_client, model_id, image_prompt, img_base64)
        
        # Overall page summary
        overall_prompt = f"""Provide a comprehensive summary of this document page:

Text Content: {text_content[:1000]}
Number of Tables: {table_count}
Table Summaries: {'; '.join(table_summaries) if table_summaries else 'None'}
Image Analysis: {image_analysis}

Please provide a detailed summary (4-5 sentences) covering the main content, key information, and any important data points."""
        
        overall_summary = analyze_with_bedrock(bedrock_client, model_id, overall_prompt)
        
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

# Main app
st.title("📄 Simple PDF Document Processor")
st.markdown("Extract, analyze, and query PDF documents with AWS Bedrock")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # AWS Bedrock settings
    st.subheader("🤖 AWS Bedrock Settings")
    aws_region = st.selectbox(
        "AWS Region",
        ["ap-south-1", "us-east-1", "us-west-2", "eu-west-1"],
        help="Select your AWS region"
    )
    
    model_id = st.selectbox(
        "Bedrock Model",
        [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0"
        ],
        help="Select a model that's enabled in your AWS Bedrock console"
    )
    
    st.info("💡 Make sure your AWS credentials are set in environment variables")
    
    # Metadata configuration
    st.subheader("📋 Document Metadata")
    
    # Default metadata fields
    ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL, MSFT", help="Stock ticker symbol")
    quarter = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"], help="Financial quarter")
    year = st.selectbox("Year", list(range(2020, 2025)), index=4, help="Year")
    
    # Custom metadata
    st.write("**Custom Metadata:**")
    custom_metadata = {}
    
    # Add custom metadata fields
    if "custom_metadata_count" not in st.session_state:
        st.session_state.custom_metadata_count = 0
    
    # Display existing custom metadata
    for i in range(st.session_state.custom_metadata_count):
        col1, col2 = st.columns([3, 1])
        with col1:
            key = st.text_input(f"Key {i+1}", key=f"custom_key_{i}", placeholder="e.g., Company, Sector")
        with col2:
            value = st.text_input(f"Value {i+1}", key=f"custom_value_{i}", placeholder="Value")
        if key and value:
            custom_metadata[key] = value
    
    # Add new metadata field button
    if st.button("➕ Add Custom Field", key="add_metadata"):
        st.session_state.custom_metadata_count += 1
        st.rerun()
    
    # Remove metadata field button
    if st.session_state.custom_metadata_count > 0 and st.button("➖ Remove Last Field", key="remove_metadata"):
        st.session_state.custom_metadata_count -= 1
        st.rerun()

# Main tabs
tab1, tab2, tab3 = st.tabs(["📄 Process PDF", "❓ Ask Questions", "🔍 Search by Metadata"])

with tab1:
    st.header("📄 Upload and Process PDF")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF file to process"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size = len(uploaded_file.getvalue()) / 1024
        st.info(f"📁 File: {uploaded_file.name} ({file_size:.1f} KB)")
        
        # Check AWS credentials
        if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
            st.error("❌ AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
            st.stop()
        
        # Process button
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            try:
                # Initialize components
                with st.spinner("Initializing components..."):
                    bedrock_client = initialize_bedrock(aws_region, model_id)
                    chroma_client, collection = initialize_chroma()
                    multivector_retriever, vectorstore, byte_store = initialize_multivector_retriever()
                    
                    if not bedrock_client or not collection or not multivector_retriever:
                        st.error("Failed to initialize components")
                        st.stop()
                    
                    st.session_state.chroma_client = chroma_client
                    st.session_state.collection = collection
                    st.session_state.multivector_retriever = multivector_retriever
                    st.session_state.vectorstore = vectorstore
                    st.session_state.byte_store = byte_store
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name
                
                try:
                    # Process the document
                    with st.spinner("Processing document..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Open PDF with PDF Plumber
                        with pdfplumber.open(pdf_path) as pdf:
                            total_pages = len(pdf.pages)
                            results = []
                            
                            for page_num in range(total_pages):
                                page = pdf.pages[page_num]
                                
                                # Process page
                                result = process_pdf_page(page, page_num, bedrock_client, model_id)
                                results.append(result)
                                
                                # Update progress
                                progress = (page_num + 1) / total_pages
                                progress_bar.progress(progress)
                                status_text.text(f"Processing page {page_num + 1}/{total_pages}")
                            
                            # Prepare metadata
                            base_metadata = {
                                "page_number": result["page_number"],
                                "text_word_count": result["text_word_count"],
                                "table_count": result["table_count"],
                                "filename": uploaded_file.name,
                                "ticker": ticker if ticker else "",
                                "quarter": quarter,
                                "year": year
                            }
                            
                            # Add custom metadata
                            base_metadata.update(custom_metadata)
                            
                            # Prepare documents for MultiVectorRetriever
                            parent_docs = []
                            child_docs = []
                            doc_ids = []
                            embeddings_data = {}
                            
                            for result in results:
                                # Generate unique document ID
                                doc_id = str(uuid.uuid4())
                                doc_ids.append(doc_id)
                                
                                # Create parent document (full content)
                                parent_doc = Document(
                                    page_content=result["text_content"],
                                    metadata=base_metadata.copy()
                                )
                                parent_docs.append(parent_doc)
                                
                                # Create child documents (summaries and chunks)
                                child_docs.extend([
                                    Document(page_content=result["overall_summary"], metadata={**base_metadata.copy(), "content_type": "summary"}),
                                    Document(page_content=f"Tables: {'; '.join(result['table_summaries'])}", metadata={**base_metadata.copy(), "content_type": "tables"}),
                                    Document(page_content=result["image_analysis"], metadata={**base_metadata.copy(), "content_type": "image_analysis"})
                                ])
                                
                                # Store in session state for later retrieval
                                multivector_entry = {
                                    "page_number": result["page_number"],
                                    "text_content": result["text_content"],
                                    "overall_summary": result["overall_summary"],
                                    "table_summaries": result["table_summaries"],
                                    "image_analysis": result["image_analysis"],
                                    "metadata": base_metadata.copy(),
                                    "doc_id": doc_id
                                }
                                
                                doc_key = f"{uploaded_file.name}_page_{result['page_number']}"
                                st.session_state.multivector_data[doc_key] = multivector_entry
                                embeddings_data[doc_id] = multivector_entry
                            
                            # Store in MultiVectorRetriever
                            if st.session_state.multivector_retriever:
                                # Add parent documents to byte_store
                                st.session_state.multivector_retriever.byte_store.mset(
                                    list(zip(doc_ids, parent_docs))
                                )
                                
                                # Add child documents to vectorstore
                                st.session_state.multivector_retriever.vectorstore.add_documents(child_docs)
                            
                            # Store in ChromaDB (legacy)
                            for result in results:
                                search_text = f"""
                                Page {result['page_number'] + 1}: {result['overall_summary']}
                                Text words: {result['text_word_count']}
                                Tables: {result['table_count']}
                                Table summaries: {'; '.join(result['table_summaries'])}
                                Image analysis: {result['image_analysis']}
                                """
                                
                                doc_id = f"{uploaded_file.name}_page_{result['page_number']}"
                                collection.add(
                                    documents=[search_text],
                                    metadatas=[base_metadata],
                                    ids=[doc_id]
                                )
                            
                            # Save data to pickle files
                            save_multivector_data(st.session_state.multivector_data, uploaded_file.name)
                            save_embeddings_to_pickle(embeddings_data, uploaded_file.name)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Processing completed!")
                    
                    st.success(f"🎉 Document processed successfully! {len(results)} pages analyzed and stored in ChromaDB.")
                    
                    # Show collection information
                    st.info(f"📦 **Documents stored in ChromaDB Collection:** `{st.session_state.collection.name}`")
                    st.info(f"🔗 **MultiVectorRetriever Collection:** `multivector_documents`")
                    
                    # Display results
                    st.header("📊 Analysis Results")
                    
                    for result in results:
                        with st.expander(f"Page {result['page_number'] + 1}", expanded=result['page_number'] < 3):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Text Words", result['text_word_count'])
                                st.metric("Tables", result['table_count'])
                                
                                if result.get('error'):
                                    st.error("❌ Processing failed")
                                else:
                                    st.success("✅ Processing successful")
                            
                            with col2:
                                st.subheader("📝 Overall Summary")
                                st.write(result['overall_summary'])
                                
                                if result['table_summaries']:
                                    st.subheader("📊 Table Summaries")
                                    for i, summary in enumerate(result['table_summaries']):
                                        st.write(f"**Table {i+1}:** {summary}")
                                
                                st.subheader("🖼️ Image Analysis")
                                st.write(result['image_analysis'])
                    
                    # Performance summary
                    st.header("📈 Performance Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Pages", len(results))
                    
                    with col2:
                        success_count = len([r for r in results if not r.get('error')])
                        st.metric("Success Rate", f"{success_count}/{len(results)}")
                    
                    with col3:
                        total_words = sum(r['text_word_count'] for r in results)
                        total_tables = sum(r['table_count'] for r in results)
                        st.metric("Total Content", f"{total_words} words, {total_tables} tables")
                
                finally:
                    # Cleanup
                    os.unlink(pdf_path)
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                st.exception(e)

with tab2:
    st.header("❓ Ask Questions About Your Documents")
    
    if st.session_state.multivector_retriever is None:
        st.info("📄 Please process a PDF document first to enable Q&A functionality.")
        # Load existing multivector data
        if not st.session_state.multivector_data:
            st.info("🔄 Loading existing document data...")
            # Try to load from pickle files
            pickle_files = [f for f in os.listdir('.') if f.startswith('multivector_data_') and f.endswith('.pkl')]
            for pickle_file in pickle_files:
                filename = pickle_file.replace('multivector_data_', '').replace('.pkl', '')
                loaded_data = load_multivector_data(filename)
                st.session_state.multivector_data.update(loaded_data)
    else:
        # Load existing data if not already loaded
        if not st.session_state.multivector_data:
            st.info("🔄 Loading existing document data...")
            # Try to load from pickle files
            pickle_files = [f for f in os.listdir('.') if f.startswith('multivector_data_') and f.endswith('.pkl')]
            for pickle_file in pickle_files:
                filename = pickle_file.replace('multivector_data_', '').replace('.pkl', '')
                loaded_data = load_multivector_data(filename)
                st.session_state.multivector_data.update(loaded_data)
        
        # Metadata filtering section
        st.subheader("🔍 Filter by Metadata (Optional)")
        
        # Collection filter
        st.write("**📦 ChromaDB Collection Filter:**")
        available_collections = get_all_chroma_collections()
        if available_collections:
            selected_collection = st.selectbox(
                "Select ChromaDB Collection", 
                ["All Collections"] + available_collections,
                help="Filter by specific ChromaDB collection"
            )
        else:
            selected_collection = "All Collections"
            st.info("No ChromaDB collections found")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Document Filters:**")
            
            # Get unique values for filters
            all_tickers = ["All"] + list(set(data.get('metadata', {}).get('ticker', '') for data in st.session_state.multivector_data.values() if data.get('metadata', {}).get('ticker')))
            all_quarters = ["All"] + list(set(data.get('metadata', {}).get('quarter', '') for data in st.session_state.multivector_data.values() if data.get('metadata', {}).get('quarter')))
            all_years = ["All"] + list(set(str(data.get('metadata', {}).get('year', '')) for data in st.session_state.multivector_data.values() if data.get('metadata', {}).get('year')))
            
            selected_ticker = st.selectbox("Ticker Symbol", all_tickers, help="Filter by specific ticker")
            selected_quarter = st.selectbox("Quarter", all_quarters, help="Filter by specific quarter")
            selected_year = st.selectbox("Year", all_years, help="Filter by specific year")
        
        with col2:
            st.write("**🔧 Custom Metadata Filter:**")
            custom_key = st.text_input("Metadata Key", placeholder="e.g., Company, Sector")
            custom_value = st.text_input("Metadata Value", placeholder="Value to search for")
        
        # Question input
        st.subheader("❓ Ask Your Question")
        question = st.text_input(
            "Ask a question about your filtered documents:",
            placeholder="e.g., What tables are in the document? What are the key findings?",
            help="Ask questions about the processed documents (filtered by metadata if specified)"
        )
        
        # Number of results
        num_results = st.slider("Number of Results", 1, 20, 5, help="Number of relevant document chunks to retrieve")
        
        if question:
            if st.button("🔍 Search & Answer", type="primary", use_container_width=True):
                try:
                    # Filter documents by metadata
                    filtered_data = {}
                    for doc_key, data in st.session_state.multivector_data.items():
                        metadata = data.get('metadata', {})
                        
                        # Apply filters
                        if selected_ticker != "All" and metadata.get('ticker') != selected_ticker:
                            continue
                        if selected_quarter != "All" and metadata.get('quarter') != selected_quarter:
                            continue
                        if selected_year != "All" and str(metadata.get('year', '')) != selected_year:
                            continue
                        if custom_key and custom_value and metadata.get(custom_key) != custom_value:
                            continue
                        
                        filtered_data[doc_key] = data
                    
                    if filtered_data:
                        # Use MultiVectorRetriever for search
                        if st.session_state.multivector_retriever:
                            with st.spinner("Searching and generating answer..."):
                                # Get relevant documents using MultiVectorRetriever
                                results = st.session_state.multivector_retriever.get_relevant_documents(
                                    question,
                                    k=num_results
                                )
                                
                                if results:
                                    st.subheader("🔍 Search Results")
                                    
                                    # Display individual results
                                    for i, doc in enumerate(results):
                                        with st.expander(f"Result {i+1}", expanded=i < 2):
                                            st.write("**Content:**")
                                            st.write(doc.page_content)
                                            
                                            st.write("**Metadata:**")
                                            for key, value in doc.metadata.items():
                                                st.write(f"- {key}: {value}")
                                    
                                    # Generate answer using Bedrock
                                    combined_content = "\n\n".join([doc.page_content for doc in results])
                                    answer_prompt = f"""Based on the following filtered document content, provide a direct and specific answer to this question: "{question}"

Document Content:
{combined_content}

Please provide:
1. A direct answer to the question
2. Specific data points or facts that support your answer
3. The metadata information where this data was found (ticker, quarter, year, etc.)

Answer:"""
                                    
                                    with st.spinner("Generating answer..."):
                                        bedrock_client = initialize_bedrock(aws_region, model_id)
                                        if bedrock_client:
                                            direct_answer = analyze_with_bedrock(bedrock_client, model_id, answer_prompt)
                                            
                                            st.subheader("💡 Direct Answer")
                                            st.write(direct_answer)
                                    
                                    # Show detailed source context
                                    st.subheader("📄 Source Context")
                                    for i, doc in enumerate(results):
                                        with st.expander(f"Source {i+1} - {doc.metadata.get('content_type', 'Unknown')}", expanded=i < 2):
                                            st.write("**Content:**")
                                            st.write(doc.page_content)
                                            
                                            st.write("**Metadata:**")
                                            for key, value in doc.metadata.items():
                                                st.write(f"- {key}: {value}")
                                            
                                            # Show additional context from multivector data
                                            doc_id = doc.metadata.get('doc_id')
                                            if doc_id:
                                                for mv_key, mv_data in st.session_state.multivector_data.items():
                                                    if mv_data.get('doc_id') == doc_id:
                                                        if mv_data.get("table_summaries"):
                                                            st.write("**Table Summaries:**")
                                                            for j, summary in enumerate(mv_data["table_summaries"]):
                                                                st.write(f"- Table {j+1}: {summary}")
                                                        
                                                        if mv_data.get("image_analysis"):
                                                            st.write("**Image Analysis:**")
                                                            st.write(mv_data["image_analysis"])
                                                        break
                                else:
                                    st.info("No relevant results found for the given question and filters. Try rephrasing your question or adjusting the filters.")
                        else:
                            st.error("MultiVectorRetriever not available")
                    else:
                        st.warning("No documents match the selected filters. Please adjust your filter criteria.")
                
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.exception(e)

with tab3:
    st.header("🔍 Search by Metadata")
    
    if st.session_state.multivector_retriever is None:
        st.info("📄 Please process a PDF document first to enable metadata search.")
    else:
        # Load existing data
        if not st.session_state.multivector_data:
            st.info("🔄 Loading existing document data...")
            pickle_files = [f for f in os.listdir('.') if f.startswith('multivector_data_') and f.endswith('.pkl')]
            for pickle_file in pickle_files:
                filename = pickle_file.replace('multivector_data_', '').replace('.pkl', '')
                loaded_data = load_multivector_data(filename)
                st.session_state.multivector_data.update(loaded_data)
        
        # Metadata search options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Filter by Metadata")
            
            # Collection filter
            st.write("**📦 ChromaDB Collection Filter:**")
            available_collections = get_all_chroma_collections()
            if available_collections:
                selected_collection = st.selectbox(
                    "Select ChromaDB Collection", 
                    ["All Collections"] + available_collections,
                    help="Filter by specific ChromaDB collection"
                )
            else:
                selected_collection = "All Collections"
                st.info("No ChromaDB collections found")
            
            # Get unique values for filters
            all_metadata = [data.get('metadata', {}) for data in st.session_state.multivector_data.values()]
            
            # Ticker filter
            tickers = list(set([m.get('ticker', '') for m in all_metadata if m.get('ticker')]))
            selected_ticker = st.selectbox("Ticker Symbol", ["All"] + tickers)
            
            # Quarter filter
            quarters = list(set([m.get('quarter', '') for m in all_metadata if m.get('quarter')]))
            selected_quarter = st.selectbox("Quarter", ["All"] + quarters)
            
            # Year filter
            years = list(set([m.get('year', '') for m in all_metadata if m.get('year')]))
            selected_year = st.selectbox("Year", ["All"] + years)
            
            # Content type filter
            content_types = ["All", "summary", "tables", "image_analysis", "full_text"]
            selected_content_type = st.selectbox("Content Type", content_types)
            
            # Custom metadata filter
            st.write("**Custom Metadata Filter:**")
            custom_key = st.text_input("Metadata Key", placeholder="e.g., Company, Sector")
            custom_value = st.text_input("Metadata Value", placeholder="Value to search for")
        
        with col2:
            st.subheader("🔍 Search Options")
            
            # Search query
            search_query = st.text_input(
                "Search Query",
                placeholder="Enter your search query",
                help="Search within filtered documents"
            )
            
            # Number of results
            num_results = st.slider("Number of Results", 1, 20, 5)
            
            # Search button
            if st.button("🔍 Search", type="primary", use_container_width=True):
                if search_query:
                    try:
                        # Filter documents by metadata
                        filtered_data = {}
                        for doc_key, data in st.session_state.multivector_data.items():
                            metadata = data.get('metadata', {})
                            
                            # Apply filters
                            if selected_ticker != "All" and metadata.get('ticker') != selected_ticker:
                                continue
                            if selected_quarter != "All" and metadata.get('quarter') != selected_quarter:
                                continue
                            if selected_year != "All" and metadata.get('year') != selected_year:
                                continue
                            if custom_key and custom_value and metadata.get(custom_key) != custom_value:
                                continue
                            
                            filtered_data[doc_key] = data
                        
                        if filtered_data:
                            # Use MultiVectorRetriever for search
                            if st.session_state.multivector_retriever:
                                with st.spinner("Searching..."):
                                    # Create a filtered retriever
                                    results = st.session_state.multivector_retriever.get_relevant_documents(
                                        search_query,
                                        k=num_results
                                    )
                                    
                                    if results:
                                        st.subheader("🔍 Search Results")
                                        
                                        for i, doc in enumerate(results):
                                            with st.expander(f"Result {i+1}", expanded=i < 2):
                                                st.write("**Content:**")
                                                st.write(doc.page_content)
                                                
                                                st.write("**Metadata:**")
                                                for key, value in doc.metadata.items():
                                                    st.write(f"- {key}: {value}")
                                        
                                        # Generate answer using Bedrock
                                        combined_content = "\n\n".join([doc.page_content for doc in results])
                                        answer_prompt = f"""Based on the following filtered document content, provide a direct answer to this question: "{search_query}"

Document Content:
{combined_content}

Please provide:
1. A direct answer to the question
2. Specific data points or facts that support your answer
3. The metadata information where this data was found

Answer:"""
                                        
                                        with st.spinner("Generating answer..."):
                                            bedrock_client = initialize_bedrock(aws_region, model_id)
                                            if bedrock_client:
                                                direct_answer = analyze_with_bedrock(bedrock_client, model_id, answer_prompt)
                                                
                                                st.subheader("💡 Direct Answer")
                                                st.write(direct_answer)
                                    else:
                                        st.info("No results found for the given query and filters.")
                            else:
                                st.error("MultiVectorRetriever not available")
                        else:
                            st.warning("No documents match the selected filters.")
                    
                    except Exception as e:
                        st.error(f"Search failed: {e}")
        
        # Show available metadata
        if st.session_state.multivector_data:
            st.subheader("📊 Available Documents and Metadata")
            
            # Group by filename
            files_metadata = {}
            for doc_key, data in st.session_state.multivector_data.items():
                filename = data.get('metadata', {}).get('filename', 'Unknown')
                if filename not in files_metadata:
                    files_metadata[filename] = []
                files_metadata[filename].append(data)
            
            for filename, file_data in files_metadata.items():
                with st.expander(f"📄 {filename} ({len(file_data)} pages)", expanded=False):
                    if file_data:
                        metadata = file_data[0]['metadata']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Basic Metadata:**")
                            if metadata.get('ticker'):
                                st.write(f"- Ticker: {metadata['ticker']}")
                            if metadata.get('quarter') and metadata.get('year'):
                                st.write(f"- Period: {metadata['quarter']} {metadata['year']}")
                            st.write(f"- Pages: {len(file_data)}")
                        
                        with col2:
                            st.write("**Content Summary:**")
                            total_words = sum(data.get('metadata', {}).get('text_word_count', 0) for data in file_data)
                            total_tables = sum(data.get('metadata', {}).get('table_count', 0) for data in file_data)
                            st.write(f"- Total Words: {total_words}")
                            st.write(f"- Total Tables: {total_tables}")
                        
                        # Show custom metadata
                        custom_fields = {k: v for k, v in metadata.items() 
                                       if k not in ['page_number', 'text_word_count', 'table_count', 
                                                   'filename', 'ticker', 'quarter', 'year']}
                        if custom_fields:
                            st.write("**Custom Metadata:**")
                            for key, value in custom_fields.items():
                                st.write(f"- {key}: {value}")

# Information section
with st.expander("ℹ️ About This Processor"):
    st.markdown("""
    **Simple PDF Document Processor**
    
    This processor uses:
    - **PDF Plumber**: Fast PDF text and table extraction
    - **AWS Bedrock**: Intelligent content analysis with Claude models
    - **ChromaDB**: Searchable storage of processed documents
    
    **Features:**
    - 📄 Text extraction and word counting
    - 📊 Table detection and summarization
    - 🖼️ Image and chart analysis
    - 🔍 Q&A functionality
    - 💾 Persistent storage in ChromaDB
    
    **Processing:**
    - Extracts text and counts words
    - Identifies and summarizes tables
    - Analyzes images and visual content
    - Provides comprehensive page summaries
    - Stores everything for easy querying
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, PDF Plumber, and AWS Bedrock") 