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
from unstructured.documents.elements import CompositeElement, NarrativeText, Image, Table
import base64

# Add langchain_aws imports
from langchain_aws import BedrockEmbeddings

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

# ------------------ SETTINGS ------------------ #
# # Get AWS Bedrock credentials from environment variable
# aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
# aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
# aws_region = os.getenv("AWS_REGION")

# bedrock = boto3.client(
#     service_name="bedrock-runtime",
#     region_name=aws_region,
#     aws_access_key_id=aws_access_key_id,
#     aws_secret_access_key=aws_secret_access_key,
# )

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

def enable_bedrock(region='ap-south-1', embedding_model_id="amazon.titan-embed-text-v2:0"):
    """
    Enable AWS Bedrock using credentials and region from environment variables.
    Returns a Bedrock runtime client and BedrockEmbeddings object.
    """
    # Set region if not already set
    if not os.environ.get('AWS_REGION'):
        os.environ['AWS_REGION'] = region

    # Create Bedrock runtime client using environment variables what
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=os.environ['AWS_REGION'],
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.environ.get('AWS_SESSION_TOKEN')  # This will be None if not set, which is fine
    )

    # Create Bedrock embeddings object (adjust model_id as needed)
    bedrock_embeddings = BedrockEmbeddings(
        model_id=embedding_model_id,
        client=bedrock_runtime
    )

    return bedrock_runtime, bedrock_embeddings

# Define model IDs at the top for global use
BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
CHAT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

bedrock_runtime, bedrock_embeddings = enable_bedrock(embedding_model_id=BEDROCK_EMBEDDING_MODEL_ID)

# Wrapper to make BedrockEmbeddings compatible with ChromaDB
class ChromaBedrockEmbeddingWrapper:
    def __init__(self, bedrock_embeddings):
        self.bedrock_embeddings = bedrock_embeddings
        self.model_id = getattr(bedrock_embeddings, "model_id", "bedrock")
    def __call__(self, input):
        # input is a list of strings
        if hasattr(self.bedrock_embeddings, "embed_documents"):
            return self.bedrock_embeddings.embed_documents(input)
        elif hasattr(self.bedrock_embeddings, "__call__"):
            return self.bedrock_embeddings(input)
        else:
            raise RuntimeError("bedrock_embeddings has no suitable embedding method (embed_documents or __call__)")
    def name(self):
        return str(self.model_id)

chroma_embedding_function = ChromaBedrockEmbeddingWrapper(bedrock_embeddings)

# Bedrock Chat Completion
def bedrock_chat(prompt, model_id=None):
    if model_id is None:
        model_id = CHAT_MODEL_ID
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
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

        # Partition PDF with robust parameters
        try:
            elements = partition_pdf(
                filename=temp_path,
                extract_images_in_pdf=True,
                infer_table_structure=True,
                strategy="hi_res",
                hi_res_model_name="yolox",
                chunking_strategy="by_title",
                # extract_image_block_types=["Image"],   
                extract_image_block_to_payload=True,  
                max_characters=4000,
                new_after_n_chars=3800,
                combine_text_under_n_chars=2000,
                image_output_dir_path='imgs',
            )
            if elements is None:
                raise ValueError(
                    "This PDF could not be processed (partition_pdf returned None). "
                    "It may be corrupted, encrypted, or unsupported. "
                    "Try opening and re-saving the PDF, or use a different file."
                )
            if not hasattr(elements, '__len__'):
                raise ValueError("partition_pdf did not return a list-like object")
        except Exception as e:
            raise ValueError(f"PDF processing failed: {str(e)}")

        if not elements:
            raise ValueError("No content could be extracted from the PDF")

        # Extract tables, texts, and images robustly
        tables = []
        texts = []
        images = []
        found_base64_set = set()
        for chunk in elements:
            try:
                if isinstance(chunk, Table):
                    tables.append({
                        'content': getattr(chunk, 'text', ''),
                        'page_number': getattr(chunk.metadata, 'page_number', 1),
                        'html': getattr(chunk.metadata, 'text_as_html', '')
                    })
                elif isinstance(chunk, Image) and hasattr(chunk.metadata, 'image_base64') and chunk.metadata.image_base64:
                    img_b64 = chunk.metadata.image_base64
                    images.append({
                        'base64': img_b64,
                        'page_number': getattr(chunk.metadata, 'page_number', 1)
                    })
                    found_base64_set.add(img_b64)
                elif isinstance(chunk, CompositeElement):
                    if hasattr(chunk, 'text') and chunk.text:
                        texts.append({
                            'content': chunk.text,
                            'page_number': getattr(chunk.metadata, 'page_number', 1)
                        })
                    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                        for el in chunk.metadata.orig_elements:
                            if isinstance(el, Image) and hasattr(el.metadata, 'image_base64') and el.metadata.image_base64:
                                img_b64 = el.metadata.image_base64
                                if img_b64 not in found_base64_set:
                                    images.append({
                                        'base64': img_b64,
                                        'page_number': getattr(chunk.metadata, 'page_number', 1)
                                    })
                                    found_base64_set.add(img_b64)
                            elif isinstance(el, Table):
                                tables.append({
                                    'content': getattr(el, 'text', ''),
                                    'page_number': getattr(el.metadata, 'page_number', 1),
                                    'html': getattr(el.metadata, 'text_as_html', '')
                                })
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
        # Fallback: read images from imgs/ directory if not already present
        img_dir = 'imgs'
        if os.path.isdir(img_dir):
            for fname in os.listdir(img_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(img_dir, fname)
                    try:
                        with open(img_path, 'rb') as f:
                            b64 = base64.b64encode(f.read()).decode('utf-8')
                            if b64 not in found_base64_set:
                                images.append({'base64': b64, 'page_number': None})
                                found_base64_set.add(b64)
                    except Exception as e:
                        print(f"Error reading image file {img_path}: {e}")
        return texts, tables, images
    except Exception as e:
        raise ValueError(f"Failed to process {uploaded_file.name}: {str(e)}")
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass  # Ignore cleanup errors

def summarize_text_with_bedrock(text_content):
    prompt = f"You are an assistant tasked with summarizing text.\nGive a concise summary of the text.\n\nRespond only with the summary, no additional comment.\nDo not start your message by saying \"Here is a summary\" or anything like that.\nJust give the summary as it is.\n\nText chunk: {text_content}"
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }
    response = bedrock_runtime.invoke_model(
        modelId=CHAT_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response['body'].read())
    return result['content'][0]['text']

def summarize_table_with_bedrock(table_html):
    prompt = f"You are an assistant tasked with summarizing tables.\nGive a concise summary of the table.\n\nRespond only with the summary, no additional comment.\nDo not start your message by saying \"Here is a summary\" or anything like that.\nJust give the summary as it is.\n\nTable HTML: {table_html}"
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }
    response = bedrock_runtime.invoke_model(
        modelId=CHAT_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response['body'].read())
    return result['content'][0]['text']

def summarize_image_with_bedrock(image_base64):
    if not image_base64 or not isinstance(image_base64, str) or image_base64.strip() == "":
        raise ValueError("Image base64 data is missing or invalid")
    prompt = f"Describe the image in detail. Be specific about any graphs, charts, or visual elements."
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
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
    }
    response = bedrock_runtime.invoke_model(
        modelId=CHAT_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
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
    st.subheader("Data Upload")
    col_left, col_center, col_right = st.columns([6, 1, 1])
    with col_left:
        # Prompt user for data store name
        data_store_name = st.text_input("Enter a data store name")
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
            f.seek(0)
            if size == 0:
                empty_files.append(f.name)
            else:
                nonempty_files.append(f)
        if empty_files:
            st.warning(f"The following files are empty and will be skipped: {', '.join(empty_files)}")

    # Section mapping UI (restored)
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
        show_process_button = True
    elif nonempty_files:
        for f in nonempty_files:
            doc_to_sections[f.name] = ["Unmapped"]
        show_process_button = True
    else:
        show_process_button = False

    process_clicked = False  # Ensure it's always defined before any conditional
    if show_process_button:
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

    if process_clicked:
        st.subheader("Extraction Preview")
        # Try to get or create the collection, handle embedding dimension errors
        try:
            collection = client.get_or_create_collection(
                name=data_store_name,
                embedding_function=chroma_embedding_function
            )
        except Exception as e:
            if "dimension" in str(e) or "embedding" in str(e):
                st.warning("Collection embedding dimension mismatch. Deleting and recreating collection...")
                try:
                    client.delete_collection(name=data_store_name)
                    collection = client.get_or_create_collection(
                        name=data_store_name,
                        embedding_function=chroma_embedding_function
                    )
                    st.success("Collection recreated successfully.")
                except Exception as e2:
                    st.error(f"Failed to recreate collection: {e2}")
                    st.stop()
            else:
                st.error(f"Failed to get or create collection: {e}")
                st.stop()
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
                    if not image.get('base64') or not isinstance(image['base64'], str) or image['base64'].strip() == "":
                        st.warning(f"Skipping image on page {image.get('page_number', '?')} due to missing/invalid base64 data.")
                        continue
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
        st.success(f"Data store created: {data_store_name}")
        st.info("Move to the next tab to generate the report.")
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
    # Use the generated data_store_name by default
    if data_store_name:
        col_info = get_collection_info(data_store_name)
        st.info(f"Collection: {data_store_name} | Model: {col_info.get('model', 'Unknown')}")
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
                        name=data_store_name,
                        embedding_function=chroma_embedding_function
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
        st.info("No data store available. Please process some documents first.")

with tab3:
    st.header("Validation")
    st.subheader("Current Configuration")
    st.write(f"**Embedding Model:** {bedrock_embeddings.model_id}")
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

