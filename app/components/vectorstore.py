from langchain_community.vectorstores import FAISS

from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

from app.config.config import DB_FAISS_PATH

import os

## To load existing vectorstore
def load_vector_store():
  try:
    embedding_model = get_embedding_model()

    if os.path.exists(DB_FAISS_PATH):
      logger.info("Loading existing vectorstore")
      return FAISS.load_local(DB_FAISS_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
    else:
      logger.warning("No vectorstore found")
  
  except Exception as e:
    error_message = CustomException("failed to load vectorstore", e)
    logger.error(str(e))

## To create the new vectorstore
def save_vector_store(text_chunks):
  try:
    if not text_chunks:
      raise CustomException("No text chunks were found")
    logger.info("Generating vector store")

    embedding_model = get_embedding_model()
    db = FAISS.from_documents(text_chunks, embedding_model)
    logger.info("saving vectorestore")
    
    db.save_local(DB_FAISS_PATH)
    logger.info("saved vectorestore successfully")

  except Exception as e:
    error_message = CustomException("failed to save vectorstore", e)
    logger.error(str(e))








