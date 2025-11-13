# from langchain_community.llms import HuggingFaceHub
# from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from app.config.config import GROQ_API_KEY, MODEL_NAME

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(groq_api_key: str = GROQ_API_KEY, model: str = MODEL_NAME):
  try:
    logger.info("Loading LLM from HuggingFace")

    llm = ChatGroq(
      api_key=groq_api_key,
      model=model,
      temperature = 0.3,
    )

    logger.info("SUCCESS loaded LLM from HuggingFace")
    return llm

  except Exception as e:
    error_message = CustomException("Error loading LLM", e)
    logger.error(str(error_message))
    return None




