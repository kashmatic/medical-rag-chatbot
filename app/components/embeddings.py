from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

## To convert our text chunks to embeddings
def get_embedding_model():
  try:
    logger.info("Initializing our Huggingface embeddings model")
    model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Huggingface embeddings model loaded")

    return model
  except Exception as e:
    logger.error(str(e))
    raise CustomException("Failed to generate model")

