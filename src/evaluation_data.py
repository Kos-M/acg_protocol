import asyncio
import json
import os
import re
from typing import List, Dict, Any

from mongodb_client import MongoDBClient
from ugvp_protocol import UGVPProtocol
from agent import GroundingAgent
from config import log
from ragas_evaluator import run_ragas_evaluation 

DEFAULT_DATASET_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "dataset.json"))

def load_dataset(path: str = DEFAULT_DATASET_PATH) -> Dict[str, Any]:
    """
    Loads the benchmark dataset (questions and optional human-written references) from JSON.

    @param path: Path to the dataset file.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)

async def get_evaluation_data(samples: List[Dict[str, str]], strategy: str = 'agent') -> List[Dict[str, Any]]:
    """
    Generates a dataset for RAGAS evaluation by querying the GroundingAgent
    with the dataset questions and extracting grounded context and agent responses.

    @param samples: Dataset samples, each with a 'question' and an optional 'reference'.
    @param strategy: The GroundingAgent generation strategy ('agent' or 'llm').
    """
    db_client = MongoDBClient()
    ugvp_instance = UGVPProtocol()
    grounding_agent = GroundingAgent(ugvp_instance, strategy=strategy)

    evaluation_data = []

    for sample in samples:
        question = sample["question"]
        log.info(f"Generating evaluation data for question: '{question}'")
        agent_response_with_acg, verified_contexts = await grounding_agent.generate_and_verify(question)
        agent_answer = re.sub(r"--- ACG_START ---.*?--- ACG_END ---", "", agent_response_with_acg, flags=re.DOTALL).strip()
        
        if agent_answer.startswith("❌ Agent unable to retrieve"):
            log.warning(f"Agent returned an error message for question '{question}' due to no grounding context. Skipping this question.")
            continue

        row = {
            "user_input": question,
            "retrieved_contexts": verified_contexts,
            "response": agent_answer,
        }
        # Only human-written references are used; scoring the agent against its own answer is meaningless.
        if sample.get("reference"):
            row["reference"] = sample["reference"]
        evaluation_data.append(row)
    
    try:
        return evaluation_data
    finally:
        db_client.close()

if __name__ == "__main__":
    async def main():
        data = await get_evaluation_data(load_dataset()["samples"])
        if data: 
            await run_ragas_evaluation(data)
        else:
            log.info("No evaluation data generated, skipping RAGAS evaluation.")
        
        try:
            import litellm
            if hasattr(litellm, 'finish') and callable(litellm.finish):
                log.info("Calling litellm.finish() to close all LiteLLM clients.")
                await litellm.finish()
        except ImportError:
            log.warning("litellm not imported, cannot call litellm.finish().")
        except Exception as e:
            log.error(f"Error calling litellm.finish(): {e}", exc_info=True)


    asyncio.run(main()) 
