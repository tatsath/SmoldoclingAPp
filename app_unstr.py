import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import uuid
import os
import time
import tempfile
from dotenv import load_dotenv
import re
import shutil
import boto3
import json
from pathlib import Path
import pandas as pd

# Add Unstructured imports
from unstructured.partition.pdf import partition_pdf
import base64

# Set wide layout
st.set_page_config(layout="wide")
# Restore previous blue panel (Kulla, not fixed, margin below)
st.markdown('<div style="background-color:#1976d2;padding:8px 0 8px 0;margin-bottom:12px;width:100%;text-align:center;color:white;font-weight:bold;font-size:26px;letter-spacing:1px;">ECC Memo Generator</div>', unsafe_allow_html=True)

# st.title("ECC Memo Generator")

# Define sections and sample prompts
SECTIONS = [
    {"name": "Transaction Overview", "sample_prompt": "Summarize the customer overview from the document."},
    {"name": "Company Overview", "sample_prompt": "List the main competitors mentioned in the document."},
    {"name": "Transaction Details", "sample_prompt": "Describe the climate-related information in the document."}
]

# Load environment variables
load_dotenv()

# ------------------ MODEL CONFIGURATIONS ------------------ #
TITAN_V2_CONFIG = {
    "modelId": "amazon.titan-embed-text-v2:0",
    "contentType": "application/json",
    "accept": "*/*",
    "body_template": {
        "inputText": "",
        "dimensions": 512,
        "normalize": True
    }
}

CLAUDE_HAIKU_CONFIG = {
    "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
    "contentType": "application/json",
    "accept": "application/json",
    "body_template": {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": ""
                    }
                ]
            }
        ]
    }
}

# ------------------ SETTINGS ------------------ #
# Get AWS Bedrock credentials from environment variable
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# Bedrock Embedding Function for ChromaDB
class BedrockEmbeddingFunction:
    def __init__(self, model_id=TITAN_V2_CONFIG["modelId"]):
        self.model_id = model_id
        self.dim = TITAN_V2_CONFIG["body_template"]["dimensions"]
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            body = TITAN_V2_CONFIG["body_template"].copy()
            body["inputText"] = text
            response = bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                accept=TITAN_V2_CONFIG["accept"],
                contentType=TITAN_V2_CONFIG["contentType"]
            )
            result = json.loads(response['body'].read())
            embeddings.append(result['embedding'])
        return embeddings
    def name(self):
        return self.model_id

embed_function = BedrockEmbeddingFunction()

# Initialize ChromaDB client with new configuration
CHROMA_DB_PATH = "./chroma_titan_v2"  # Changed to reflect the model being used
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

artifacts_path = "/Users/tatsa/.cache/docling/models"
pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# Bedrock Chat Completion
def bedrock_chat(prompt, model_id=CLAUDE_HAIKU_CONFIG["modelId"]):
    body = CLAUDE_HAIKU_CONFIG["body_template"].copy()
    body["messages"][0]["content"][0]["text"] = prompt
    
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept=CLAUDE_HAIKU_CONFIG["accept"],
        contentType=CLAUDE_HAIKU_CONFIG["contentType"]
    )
    result = json.loads(response['body'].read())
    return result['content'][0]['text']

# ------------------ UTILS ------------------ #
def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def parse_markdown_sections(markdown_text):
    sections = []
    current_section = {"title": "Introduction", "content": ""}
    
    # Split by headers (## or ###)
    lines = markdown_text.split('\n')
    for line in lines:
        if line.startswith('## '):
            # Save previous section if it has content
            if current_section["content"].strip():
                sections.append(current_section)
            # Start new section
            current_section = {
                "title": line[3:].strip(),
                "content": ""
            }
        elif line.startswith('### '):
            # Save previous section if it has content
            if current_section["content"].strip():
                sections.append(current_section)
            # Start new section
            current_section = {
                "title": line[4:].strip(),
                "content": ""
            }
        else:
            current_section["content"] += line + "\n"
    
    # Add the last section if it has content
    if current_section["content"].strip():
        sections.append(current_section)
    
    return sections

def process_file_with_unstructured(uploaded_file):
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        uploaded_file.seek(0)
        file_content = uploaded_file.read()
        if not file_content:
            raise ValueError(f"Uploaded file {uploaded_file.name} has no content")
        temp_file.write(file_content)
        temp_path = temp_file.name

    try:
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Temporary file {temp_path} was not created")
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise ValueError(f"File is empty (size: {file_size} bytes)")
        with open(temp_path, 'rb') as f:
            header = f.read(5)
            if not header.startswith(b'%PDF-'):
                raise ValueError("File does not appear to be a valid PDF (missing PDF header)")

        # Defensive: check all required params
        try:
            chunks = partition_pdf(
                filename=temp_path,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000,
            )
            if chunks is None:
                raise ValueError(
                    "This PDF could not be processed (partition_pdf returned None). "
                    "It may be corrupted, encrypted, or unsupported. "
                    "Try opening and re-saving the PDF, or use a different file."
                )
            if not hasattr(chunks, '__len__'):
                raise ValueError("partition_pdf did not return a list-like object")
        except Exception as e:
            raise ValueError(f"PDF processing failed: {str(e)}")

        if not chunks:
            raise ValueError("No content could be extracted from the PDF")

        # Separate tables, texts, and images
        tables = []
        texts = []
        images = []
        
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append({
                    'content': chunk.text,
                    'page_number': getattr(chunk.metadata, 'page_number', 1),
                    'html': getattr(chunk.metadata, 'text_as_html', '')
                })
            if "CompositeElement" in str(type(chunk)):
                texts.append({
                    'content': chunk.text,
                    'page_number': getattr(chunk.metadata, 'page_number', 1)
                })
                # Extract images from CompositeElement
                if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                    for el in chunk.metadata.orig_elements:
                        if "Image" in str(type(el)) and hasattr(el.metadata, 'image_base64'):
                            images.append({
                                'base64': el.metadata.image_base64,
                                'page_number': getattr(chunk.metadata, 'page_number', 1)
                            })
        
        return texts, tables, images
    except Exception as e:
        raise ValueError(f"Failed to process {uploaded_file.name}: {str(e)}")
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass  # Ignore cleanup errors

def summarize_text_with_bedrock(text_content):
    """Summarize text using AWS Bedrock"""
    prompt = f"""You are an assistant tasked with summarizing text.
Give a concise summary of the text.

Respond only with the summary, no additional comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Text chunk: {text_content}"""
    
    body = CLAUDE_HAIKU_CONFIG["body_template"].copy()
    body["messages"][0]["content"][0]["text"] = prompt
    
    response = bedrock.invoke_model(
        modelId=CLAUDE_HAIKU_CONFIG["modelId"],
        body=json.dumps(body),
        accept=CLAUDE_HAIKU_CONFIG["accept"],
        contentType=CLAUDE_HAIKU_CONFIG["contentType"]
    )
    result = json.loads(response['body'].read())
    return result['content'][0]['text']

def summarize_table_with_bedrock(table_html):
    """Summarize table using AWS Bedrock"""
    prompt = f"""You are an assistant tasked with summarizing tables.
Give a concise summary of the table.

Respond only with the summary, no additional comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table HTML: {table_html}"""
    
    body = CLAUDE_HAIKU_CONFIG["body_template"].copy()
    body["messages"][0]["content"][0]["text"] = prompt
    
    response = bedrock.invoke_model(
        modelId=CLAUDE_HAIKU_CONFIG["modelId"],
        body=json.dumps(body),
        accept=CLAUDE_HAIKU_CONFIG["accept"],
        contentType=CLAUDE_HAIKU_CONFIG["contentType"]
    )
    result = json.loads(response['body'].read())
    return result['content'][0]['text']

def summarize_image_with_bedrock(image_base64):
    """Summarize image using AWS Bedrock"""
    prompt = f"""Describe the image in detail. Be specific about any graphs, charts, or visual elements."""
    
    body = CLAUDE_HAIKU_CONFIG["body_template"].copy()
    body["messages"][0]["content"] = [
        {
            "type": "text",
            "text": prompt
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_base64
            }
        }
    ]
    
    response = bedrock.invoke_model(
        modelId=CLAUDE_HAIKU_CONFIG["modelId"],
        body=json.dumps(body),
        accept=CLAUDE_HAIKU_CONFIG["accept"],
        contentType=CLAUDE_HAIKU_CONFIG["contentType"]
    )
    result = json.loads(response['body'].read())
    return result['content'][0]['text']

def get_all_collections():
    return client.list_collections()

def get_collection_sections(collection_name):
    collection = client.get_collection(name=collection_name)
    # Get unique sections from metadata
    sections = set()
    for metadata in collection.get()["metadatas"]:
        if "section" in metadata:
            sections.add(metadata["section"])
    return sorted(list(sections))

def delete_collection(collection_name):
    """Delete a specific collection"""
    try:
        client.delete_collection(name=collection_name)
        return True
    except Exception as e:
        st.error(f"Error deleting collection: {e}")
        return False

def get_collection_info(collection_name):
    """Get information about a collection's embedding function"""
    try:
        collection = client.get_collection(name=collection_name)
        # Try to get embedding function info
        embedding_func = getattr(collection, "_embedding_function", None)
        if embedding_func:
            model_name = getattr(embedding_func, "model_name", "Unknown")
            return {"model": model_name, "exists": True}
        return {"model": "Unknown", "exists": True}
    except Exception as e:
        return {"model": "Error", "exists": False, "error": str(e)}

# Remove sidebar and move data store selection to main area (ECC Memo Generator tab)
existing_companies = get_all_collections()
company_name = None
store_mode = None

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Document Upload", "Memo Generation", "Validation", "Custom Report Generator"])

with tab1:
    st.subheader("Data Store Selection")
    store_mode = st.radio("Select Mode", ["New Data Store", "Use Existing Data Store"])
    col1, col2 = st.columns([1,1])
    with col1:
        if store_mode == "New Data Store":
            company_name = st.text_input("Enter New Company Name", key="company_name")
            if company_name:
                # Validate company name
                if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$', company_name):
                    st.error("Company name must be 3-512 characters long, start and end with alphanumeric characters, and contain only letters, numbers, dots, underscores, and hyphens.")
                    company_name = None
        elif store_mode == "Use Existing Data Store":
            if existing_companies:
                company_name = st.selectbox("Select Existing Company", [comp.name for comp in existing_companies])
            else:
                st.info("No companies found.")
        uploaded_files = st.file_uploader(
            "Upload multiple documents (PDF/TXT/IMP)",
            type=['pdf', 'txt', 'imp'],
            accept_multiple_files=True,
            key="file_uploader"
        )
    enable_section_mapping = st.checkbox("Enable Section Mapping (optional)")
    section_to_docs = {}
    doc_to_sections = {}

    # Filter out empty files before processing
    nonempty_files = []
    empty_files = []
    if uploaded_files:
        for f in uploaded_files:
            f.seek(0, 2)  # Move to end of file
            size = f.tell()
            f.seek(0)  # Reset to beginning
            if size == 0:
                st.error(f"File {f.name} is empty (0 bytes)")
                empty_files.append(f.name)
            else:
                st.info(f"File {f.name} size: {size} bytes")
                nonempty_files.append(f)
        if empty_files:
            st.warning(f"The following files are empty and will be skipped: {', '.join(empty_files)}")

    if enable_section_mapping and nonempty_files:
        st.subheader("Map Sections to Documents")
        doc_names = [f.name for f in nonempty_files]
        all_files_option = "All Files"
        doc_names_with_all = [all_files_option] + doc_names
        for section in SECTIONS:
            selected_docs = st.multiselect(
                f"Select documents for section: {section['name']}",
                doc_names_with_all,
                key=f"map_{section['name']}"
            )
            if all_files_option in selected_docs:
                selected_docs = doc_names
            section_to_docs[section["name"]] = selected_docs
            for doc in selected_docs:
                doc_to_sections.setdefault(doc, []).append(section["name"])
    elif nonempty_files:
        for f in nonempty_files:
            doc_to_sections[f.name] = ["Unmapped"]

    # Extraction preview UI
    if nonempty_files:
        # Only one beautiful blue button above the preview header, using Streamlit only
        st.markdown("""
            <style>
            .stButton > button {
                background-color: #1976d2;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 0.8em 2.5em;
                font-size: 1.2em;
                box-shadow: 0 2px 8px rgba(25, 118, 210, 0.15);
                border: none;
                transition: background 0.2s, box-shadow 0.2s;
                margin-bottom: 1.5em;
            }
            .stButton > button:hover {
                background-color: #1251a3;
                box-shadow: 0 4px 16px rgba(25, 118, 210, 0.25);
            }
            </style>
        """, unsafe_allow_html=True)
        process_clicked = st.button("Process and Store Documents")
        st.subheader("Extraction Preview")
        if process_clicked:
            # Check for existing collection and show info
            existing_collections = get_all_collections()
            collection_exists = any(coll.name == company_name for coll in existing_collections)
            if collection_exists:
                col_info = get_collection_info(company_name)
                st.warning(f"Collection '{company_name}' already exists with model: {col_info.get('model', 'Unknown')}")
                st.info("Documents will be added to the existing collection. Make sure the embedding model matches!")
            collection = client.get_or_create_collection(
                name=company_name, 
                embedding_function=embed_function
            )
            all_chunks = []
            total_files = len(nonempty_files)
            preview_progress = st.progress(0, text="Starting extraction and storage...")
            status_text = st.empty()
            success_files = []
            failed_files = []
            doc_chunks_map = {}  # Store chunks for each doc for display
            for idx, uploaded_file in enumerate(nonempty_files):
                status_text.text(f"Processing file {idx + 1} of {total_files}: {uploaded_file.name}")
                try:
                    texts, tables, images = process_file_with_unstructured(uploaded_file)
                    doc_chunks_map[uploaded_file.name] = {'texts': [], 'tables': [], 'images': []}
                    doc_sections = doc_to_sections.get(uploaded_file.name, ["Unmapped"])
                    # Process text chunks
                    for text_chunk in texts:
                        if isinstance(text_chunk, dict) and 'content' in text_chunk:
                            content = text_chunk['content'].strip()
                            if content:
                                summary = summarize_text_with_bedrock(content)
                                chunk_id = f"text_{uuid.uuid4().hex[:6]}"
                                collection.add(
                                    documents=[summary],
                                    metadatas=[{
                                        "filename": uploaded_file.name,
                                        "section": doc_sections[0],
                                        "type": "text",
                                        "page": text_chunk['page_number']
                                    }],
                                    ids=[chunk_id]
                                )
                                doc_chunks_map[uploaded_file.name]['texts'].append({
                                    'page_number': text_chunk['page_number'],
                                    'content': content,
                                    'summary': summary
                                })
                                all_chunks.append(summary)
                    # Process table chunks
                    for table in tables:
                        if isinstance(table, dict) and 'html' in table:
                            summary = summarize_table_with_bedrock(table['html'])
                            chunk_id = f"table_{uuid.uuid4().hex[:6]}"
                            collection.add(
                                documents=[summary],
                                metadatas=[{
                                    "filename": uploaded_file.name,
                                    "section": doc_sections[0],
                                    "type": "table",
                                    "page": table['page_number']
                                }],
                                ids=[chunk_id]
                            )
                            doc_chunks_map[uploaded_file.name]['tables'].append({
                                'page_number': table['page_number'],
                                'content': table['content'],
                                'html': table['html'],
                                'summary': summary
                            })
                            all_chunks.append(summary)
                    # Process image chunks
                    for image in images:
                        if isinstance(image, dict) and 'base64' in image:
                            summary = summarize_image_with_bedrock(image['base64'])
                            chunk_id = f"image_{uuid.uuid4().hex[:6]}"
                            collection.add(
                                documents=[summary],
                                metadatas=[{
                                    "filename": uploaded_file.name,
                                    "section": doc_sections[0],
                                    "type": "image",
                                    "page": image['page_number']
                                }],
                                ids=[chunk_id]
                            )
                            doc_chunks_map[uploaded_file.name]['images'].append({
                                'page_number': image['page_number'],
                                'base64': image['base64'],
                                'summary': summary
                            })
                            all_chunks.append(summary)
                    success_files.append(uploaded_file.name)
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                    failed_files.append(uploaded_file.name)
                    continue
                preview_progress.progress((idx + 1) / total_files, text=f"Finished extracting: {uploaded_file.name}")
                status_text.empty()
            preview_progress.empty()
            st.success("Task completed!")
            # Show all chunks in collapsible sections (no nested expanders)
            for doc_name, chunk_dict in doc_chunks_map.items():
                st.markdown(f"## {doc_name}")
                st.markdown(f"<span style='color:#1976d2;font-weight:bold;font-size:1.1em;'>Text Chunks: {len(chunk_dict['texts'])} | Table Chunks: {len(chunk_dict['tables'])} | Image Chunks: {len(chunk_dict['images'])}</span>", unsafe_allow_html=True)
                # Text chunks
                if chunk_dict['texts']:
                    with st.expander("Text Chunks", expanded=False):
                        for el in chunk_dict['texts']:
                            st.write(f"Type: text | Page: {el['page_number']}")
                            st.code(el['content'][:500])
                            st.markdown(f"<span style='color:#1976d2;font-weight:bold;'>Text Summary:</span> {el['summary']}", unsafe_allow_html=True)
                # Table chunks
                if chunk_dict['tables']:
                    with st.expander("Table Chunks", expanded=False):
                        for table in chunk_dict['tables']:
                            st.write(f"Type: table | Page: {table['page_number']}")
                            if 'html' in table and table['html']:
                                st.markdown(f"<div style='overflow-x:auto'>{table['html']}</div>", unsafe_allow_html=True)
                            else:
                                st.code(table['content'][:500])
                            st.markdown(f"<span style='color:#1976d2;font-weight:bold;'>Table Summary:</span> {table['summary']}", unsafe_allow_html=True)
                # Image chunks
                if chunk_dict['images']:
                    with st.expander("Image Chunks", expanded=False):
                        for image in chunk_dict['images']:
                            st.write(f"Type: image | Page: {image['page_number']}")
                            if 'base64' in image:
                                try:
                                    img_bytes = base64.b64decode(image['base64'])
                                    st.image(img_bytes, caption=f"Image on page {image['page_number']}", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error displaying image: {e}")
                            st.markdown(f"<span style='color:#1976d2;font-weight:bold;'>Image Summary:</span> {image['summary']}", unsafe_allow_html=True)

with tab2:
    st.header("ECC Memo Generation")
    existing_collections = get_all_collections()
    if existing_collections:
        selected_collection = st.selectbox("Select Collection", [coll.name for coll in existing_collections])
        if selected_collection:
            # Show collection info
            col_info = get_collection_info(selected_collection)
            st.info(f"Collection: {selected_collection} | Model: {col_info.get('model', 'Unknown')}")
            # Section dropdown and prompt
            section_names = [s["name"] for s in SECTIONS]
            selected_section = st.selectbox("Select Section (optional)", ["All Sections"] + section_names)
            default_prompt = next((s["sample_prompt"] for s in SECTIONS if s["name"] == selected_section), "Enter your prompt") if selected_section != "All Sections" else "Enter your prompt"
            prompt = st.text_area("Prompt", value=default_prompt)
            if st.button("Run Query") and prompt:
                # Only filter by section if a specific section is selected
                if selected_section and selected_section != "All Sections":
                    where_filter = {"section": selected_section}
                else:
                    where_filter = None
                with st.spinner("Searching for relevant content..."):
                    try:
                        collection = client.get_collection(
                            name=selected_collection,
                            embedding_function=embed_function
                        )
                        # Use query_texts for semantic search
                        results = collection.query(
                            query_texts=[prompt],
                            n_results=5,
                            where=where_filter,
                            include=["documents", "metadatas", "distances"]
                        )
                        
                        # Display the results
                        st.write("### Search Results")
                        for i, (doc, metadata, distance) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
                            st.write(f"**Result {i+1}** (Similarity: {1 - distance:.2f})")
                            st.write(f"Source: {metadata['filename']}")
                            st.write(f"Section: {metadata['section']}")
                            st.write(f"Content: {doc[:200]}...")
                            st.write("---")
                        
                        # Combine all relevant content for the final response
                        context = "\n\n".join(results["documents"][0])
                        
                        with st.spinner("Generating response..."):
                            # Create a more detailed prompt for Claude
                            enhanced_prompt = f"""Based on the following context, please answer the question: {prompt}

Context:
{context}

Please provide a comprehensive answer based on the context above."""
                            
                            response = bedrock_chat(enhanced_prompt)
                            st.write("### Generated Report")
                            st.write(response)
                    except Exception as e:
                        st.error(f"Query failed: {str(e)}")
                        st.error("This might be due to embedding dimension mismatch. Try deleting and recreating the collection.")
    else:
        st.info("No collections available. Please process some documents first.")

with tab3:
    st.header("Validation")
    st.subheader("Current Configuration")
    st.write(f"**Embedding Model:** {embed_function.model_id}")
    st.write(f"**ChromaDB Path:** {CHROMA_DB_PATH}")
    
    st.subheader("Collection Details")
    existing_collections = get_all_collections()
    if existing_collections:
        for coll in existing_collections:
            with st.expander(f"Collection: {coll.name}"):
                try:
                    collection = client.get_collection(name=coll.name)
                    count = collection.count()
                    st.write(f"**Document Count:** {count}")
                    
                    # Try to get sample documents
                    if count > 0:
                        sample = collection.get(limit=3)
                        st.write("**Sample Documents:**")
                        for i, doc in enumerate(sample["documents"]):
                            st.write(f"Document {i+1}: {doc[:100]}...")
                            
                    # Show embedding function info
                    embedding_func = getattr(collection, "_embedding_function", None)
                    if embedding_func:
                        st.write(f"**Embedding Function:** {type(embedding_func).__name__}")
                        if hasattr(embedding_func, "model_name"):
                            st.write(f"**Model Name:** {embedding_func.model_name}")
                    
                except Exception as e:
                    st.error(f"Error accessing collection {coll.name}: {str(e)}")
    else:
        st.info("No collections found.")

with tab4:
    st.header("Custom Report Section")
    st.info("This section will allow you to generate custom report sections. (Coming soon)")