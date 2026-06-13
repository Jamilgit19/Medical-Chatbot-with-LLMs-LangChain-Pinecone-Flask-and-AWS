from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_pdf_files(data_path):
    """Load all PDF files from a directory."""
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def text_split(documents):
    """Split documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


def download_embeddings():
    """Download and return the HuggingFace embeddings model (all-MiniLM-L6-v2, dim=384)."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings
