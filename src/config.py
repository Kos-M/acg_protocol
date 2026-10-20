import os
import logging
import sys
from dotenv import load_dotenv
from typing import Dict, Any

class Config:
    """Manages environment variables and application settings."""
    
    load_dotenv()

    # --- Database Configuration ---
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "ugvp")
    # Collection names for vector index and metadata storage
    MONGO_RAG_COLLECTION: str = "data"
    MONGO_SSR_COLLECTION: str = "ugvp_registry"
    
    # --- Protocol Constants ---
    SHI_ALGORITHM: str = "sha256"
    SHI_PREFIX_LENGTH: int = 10 # Length of SHI prefix to use in user-facing markers
    EMBEDDING_DIMENSION: int = 384 # Google's embedding dimension for 'text-embedding-001'
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001") 
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "") 
    # Gemini model used by the agent for generation and synthesis (the RAGAS judge model is fixed separately).
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gemini-3.8-flash")
    
    ACG_DELIMITERS: Dict[str, str] = {
        "CLAIM_START": "[",
        "CLAIM_END": "]",
        "CLAIM_SEP": ":",
        "RELATION_START": "(",
        "RELATION_END": ")",
        "RELATION_SEP": ":",
        "RELATION_TYPE_SEP": ":"
    }

    # --- Logging Configuration ---
    LOG_LEVEL: int = logging.DEBUG
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

    @classmethod
    def get_db_config(cls) -> Dict[str, str]:
        """Returns the database configuration as a dictionary."""
        return {
            "uri": cls.MONGO_URI,
            "db_name": cls.MONGO_DB_NAME
        }

LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "agent.log")

logging.basicConfig(
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger("pymongo").setLevel(logging.WARNING)
log = logging.getLogger('UGVP_POC')
log.setLevel(Config.LOG_LEVEL)
