import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTORSTORE_DIR, "index.faiss")

def ingest_resume(pdf_path: str):
    """
    Ingest resume PDF:
    - Load
    - Chunk
    - Embed
    - Store in FAISS
    """

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Add metadata
    for doc in documents:
        doc.metadata["source"] = os.path.basename(pdf_path)
        doc.metadata["type"] = "resume"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    clean_chunks = []
    for chunk in chunks:
        if len(chunk.page_content.strip()) < 60:
            continue
        clean_chunks.append(chunk)

    MAX_CHUNKS = 150
    if len(clean_chunks) > MAX_CHUNKS:
        clean_chunks = clean_chunks[:MAX_CHUNKS]

    if not clean_chunks:
        raise ValueError("No valid resume content found.")

    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=256
    )

    if os.path.exists(INDEX_FILE):
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(clean_chunks)
    else:
        vectorstore = FAISS.from_documents(clean_chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_DIR)

    print(f"Resume ingested: {len(clean_chunks)} chunks")

    return {
        "status": "success",
        "chunks_added": len(clean_chunks),
        "file": os.path.basename(pdf_path)
    }
