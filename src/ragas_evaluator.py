import asyncio
import json
import logging
from typing import List, Dict, Any
import os

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper 
from ragas.embeddings import LangchainEmbeddingsWrapper

from config import log, Config

log.setLevel(logging.INFO)

async def run_ragas_evaluation(data: List[Dict[str, Any]]):
    """
    Runs RAGAS evaluation on the provided dataset using Google Gemini via ChatGoogleGenerativeAI.

    @param data: A list of dictionaries, where each dictionary contains:
                 'user_input': The user query.
                 'retrieved_contexts': A list of retrieved contexts.
                 'response': The generated answer.
                 'reference': The ground truth answer (single string).
    """
    if not data:
        log.warning("No data provided for RAGAS evaluation.")
        return

    ragas_dataset = Dataset.from_list(data)

    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=Config.GOOGLE_API_KEY)
    ragas_llm = LangchainLLMWrapper(gemini_llm)
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=Config.GOOGLE_API_KEY
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings   
    context_recall.llm = ragas_llm
    context_precision.llm = ragas_llm    
    log.info("Starting RAGAS evaluation...")
    result = None
    try:
        result = evaluate(
            ragas_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        log.info("RAGAS evaluation completed.")
        log.info(result.to_pandas().to_json(orient="records", indent=2))
    except Exception as e:
        log.error(f"Error during RAGAS evaluation: {e}", exc_info=True)
    finally:
        if hasattr(gemini_llm, 'client') and hasattr(gemini_llm.client, 'close'):
            log.info("Closing ChatGoogleGenerativeAI client.")
            try:
                gemini_llm.client.close()
            except TypeError:  
                pass  
        elif hasattr(gemini_llm, 'aexit'):  
            log.info("Calling aexit on ChatGoogleGenerativeAI client.")
            await gemini_llm.aexit()

        if hasattr(gemini_embeddings, 'client') and hasattr(gemini_embeddings.client, 'close'):
            log.info("Closing GoogleGenerativeAIEmbeddings client.")
            try:
                gemini_embeddings.client.close()
            except TypeError:
                pass
        elif hasattr(gemini_embeddings, 'aexit'):
            log.info("Calling aexit on GoogleGenerativeAIEmbeddings client.")
            await gemini_embeddings.aexit()
            
    return result
