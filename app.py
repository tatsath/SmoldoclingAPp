import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import openai
import uuid
import os
import time
import tempfile
from dotenv import load_dotenv
import re
import shutil

# Define sections and sample prompts
SECTIONS = [
    {"name": "Customer Overview", "sample_prompt": "Summarize the customer overview from the document."},
    {"name": "Competitors", "sample_prompt": "List the main competitors mentioned in the document."},
    {"name": "Climate", "sample_prompt": "Describe the climate-related information in the document."}
]

# Load environment variables
load_dotenv()

# ------------------ SETTINGS ------------------ #
# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set your OpenAI API key in the .env file")
    st.info("Create a .env file in the project directory with: OPENAI_API_KEY=your_api_key_here")
    st.stop()

openai.api_key = openai_api_key

# Updated to use the latest OpenAI embedding model
embedding_model = "text-embedding-3-small"  # 1536 dimensions, same as ada-002 but newer
embed_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key, 
    model_name=embedding_model
)

# Initialize ChromaDB client with new configuration
client = chromadb.PersistentClient(path="./chroma_ada")

artifacts_path = "/Users/tatsa/.cache/docling/models"
pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

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

def process_file(uploaded_file):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    
    try:
        # Convert the document using the temporary file path
        result = doc_converter.convert(temp_path)
        # Get markdown content
        markdown_content = result.document.export_to_markdown()
        return {"content": markdown_content}
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

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

# ------------------ UI ------------------ #
st.title("Document Processing and Querying System")

# Sidebar for collection management
st.sidebar.header("Collection Management")
collection_name = st.sidebar.text_input("Enter collection name")

# Enhanced collection management
st.sidebar.subheader("ChromaDB Management")

# Button to clear ChromaDB data directory
def clear_chromadb():
    if os.path.exists("./chroma_ada"):
        shutil.rmtree("./chroma_ada")
        st.sidebar.success("ChromaDB data directory cleared. Please restart the app.")
    else:
        st.sidebar.info("ChromaDB data directory does not exist.")

if st.sidebar.button("Clear All ChromaDB Data"):
    clear_chromadb()

existing_collections = get_all_collections()
if existing_collections:
    st.sidebar.subheader("Existing Collections")
    for coll in existing_collections:
        col_info = get_collection_info(coll.name)
        st.sidebar.write(f"**{coll.name}**")
        st.sidebar.write(f"  Model: {col_info.get('model', 'Unknown')}")
        
        # Add delete button for each collection
        if st.sidebar.button(f"Delete {coll.name}", key=f"delete_{coll.name}"):
            if delete_collection(coll.name):
                st.sidebar.success(f"Collection '{coll.name}' deleted!")
                st.rerun()

# Show current embedding model info
st.sidebar.subheader("Current Settings")
st.sidebar.write(f"**Embedding Model:** {embedding_model}")
st.sidebar.write("**Dimensions:** 1536")

# Main content area
tab1, tab2, tab3 = st.tabs(["Process Documents", "Query Documents", "Debug Info"])

with tab1:
    st.header("Process and Store Documents")
    uploaded_files = st.file_uploader("Upload multiple documents (PDF/TXT/IMP)", type=['pdf', 'txt', 'imp'], accept_multiple_files=True)
    enable_section_mapping = st.checkbox("Enable Section Mapping (optional)")
    section_to_docs = {}
    doc_to_sections = {}

    if enable_section_mapping and uploaded_files:
        st.subheader("Map Sections to Documents")
        doc_names = [f.name for f in uploaded_files]
        for section in SECTIONS:
            selected_docs = st.multiselect(
                f"Select documents for section: {section['name']}",
                doc_names,
                key=f"map_{section['name']}"
            )
            section_to_docs[section["name"]] = selected_docs
            for doc in selected_docs:
                doc_to_sections.setdefault(doc, []).append(section["name"])
    elif uploaded_files:
        # If not mapping, all docs get section "Unmapped"
        for f in uploaded_files:
            doc_to_sections[f.name] = ["Unmapped"]

    if uploaded_files and collection_name:
        # Check if collection already exists and warn about embedding compatibility
        existing_collections = get_all_collections()
        collection_exists = any(coll.name == collection_name for coll in existing_collections)
        
        if collection_exists:
            col_info = get_collection_info(collection_name)
            st.warning(f"Collection '{collection_name}' already exists with model: {col_info.get('model', 'Unknown')}")
            st.info("Documents will be added to the existing collection. Make sure the embedding model matches!")
        
        if st.button("Process and Store Documents"):
            collection = client.get_or_create_collection(
                name=collection_name, 
                embedding_function=embed_function
            )
            all_chunks = []
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_text = st.empty()
            # Avoid re-embedding the same document
            embedded_docs = {}
            for idx, uploaded_file in enumerate(uploaded_files):
                doc_sections = doc_to_sections.get(uploaded_file.name, ["Unmapped"])
                status_text.text(f"Processing file {idx + 1} of {total_files}: {uploaded_file.name}")
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    start_time = time.time()
                    doc_data = process_file(uploaded_file)
                    processing_time = time.time() - start_time
                    st.info(f"Processed {uploaded_file.name} in {processing_time:.2f} seconds")
                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    # Only embed once per document
                    if uploaded_file.name not in embedded_docs:
                        if ext == ".imp":
                            embedded_docs[uploaded_file.name] = [doc_data["content"]]
                        else:
                            embedded_docs[uploaded_file.name] = chunk_text(doc_data["content"])
                    doc_chunks = embedded_docs[uploaded_file.name]
                    # Per-chunk progress
                    chunk_progress = st.progress(0, text=f"Embedding chunks for {uploaded_file.name}")
                    for i, chunk in enumerate(doc_chunks):
                        for section in doc_sections:
                            chunk_id = f"chunk_{i}_{uuid.uuid4().hex[:6]}"
                            collection.add(
                                documents=[chunk],
                                metadatas=[{"filename": uploaded_file.name, "section": section}],
                                ids=[chunk_id]
                            )
                            all_chunks.append(chunk)
                        chunk_progress.progress((i + 1) / len(doc_chunks), text=f"Embedded chunk {i+1}/{len(doc_chunks)} for {uploaded_file.name}")
                    chunk_progress.empty()
                progress_bar.progress((idx + 1) / total_files)
                status_text.empty()  # Clear after each file
            progress_bar.empty()
            status_text.empty()
            st.success(f"Stored {len(all_chunks)} chunks from {len(uploaded_files)} documents in: {collection_name}")
            st.info("All files processed!")

with tab2:
    st.header("Query Documents")
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
                    where_filter = {"section": {"eq": selected_section}}
                else:
                    where_filter = None
                with st.spinner("Searching for relevant content..."):
                    try:
                        collection = client.get_collection(
                            name=selected_collection,
                            embedding_function=embed_function
                        )
                        results = collection.query(
                            query_texts=[prompt],
                            n_results=5,
                            where=where_filter
                        )
                        context = "\n".join(results["documents"][0])
                        with st.spinner("Generating response..."):
                            from openai import OpenAI
                            client_openai = OpenAI(api_key=openai_api_key)
                            response = client_openai.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant generating reports from document context."},
                                    {"role": "user", "content": f"Use this info to generate a report:\n{context}\n\nQuery: {prompt}"}
                                ]
                            )
                            st.write("### Generated Report")
                            st.write(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Query failed: {str(e)}")
                        st.error("This might be due to embedding dimension mismatch. Try deleting and recreating the collection.")
    else:
        st.info("No collections available. Please process some documents first.")

with tab3:
    st.header("Debug Information")
    st.subheader("Current Configuration")
    st.write(f"**Embedding Model:** {embedding_model}")
    st.write(f"**ChromaDB Path:** ./chroma_ada")
    st.write(f"**OpenAI API Key Set:** {'Yes' if openai_api_key else 'No'}")
    
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