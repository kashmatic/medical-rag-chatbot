import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdf_files():
  try:
    if not os.path.exists(DATA_PATH):
      raise CustomException(f"Data path does not exist {DATA_PATH}")
    logger.info(f"Loading files form {DATA_PATH}")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)

    documents=loader.load()
    if not documents:
      logger.warning(f"No documents to load")
    else:
      logger.info(f"Number of documents loaded: {len(documents)}")

    return documents
  except Exception as e:
    logger.error(str(e))
    error_message = CustomException("Failed to load PDFs", e)
    return []

def create_text_chunks(documents):
  try:
    if not documents:
      raise CustomException("No documents found")
    logger.info(f"Splitting {len(documents)} documents into chunks")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    text_chunks = text_splitter.split_documents(documents)

    logger.info(f"Generated {len(text_chunks)} number of text chunks")

    return text_chunks
  except Exception as e:
    logger.error(str(e))
    raise CustomException("Failes to get text chunks")

