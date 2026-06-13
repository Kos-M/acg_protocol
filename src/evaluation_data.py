import asyncio
import re
from typing import List, Dict, Any

from mongodb_client import MongoDBClient
from ugvp_protocol import UGVPProtocol
from agent import GroundingAgent
from config import log
from ragas_evaluator import run_ragas_evaluation 

async def get_evaluation_data() -> List[Dict[str, Any]]:
    """
    Generates a dataset for RAGAS evaluation by querying the GroundingAgent
    with predefined questions and extracting grounded context and agent responses.
    """
    db_client = MongoDBClient()
    ugvp_instance = UGVPProtocol()
    grounding_agent = GroundingAgent(ugvp_instance, strategy='agent')

    evaluation_data = []

    # Predefined list of questions for evaluation
    questions = [
        "What was one of the earliest recorded forms of cryptographic schemes, and which ancient civilization is noted for employing it?",
        "During the Renaissance, what key vulnerability of older ciphers did the introduction of polyalphabetic ciphers, such as the Vigenère cipher, successfully address?",
        "Besides the cracking of the Enigma code, what is another historical case study mentioned in the article where cryptographic vulnerability had a significant, tangible impact on global events during World War II?",
        "The paper navigates through the evolution of cryptography, shedding light on four major modern schemes: OTP, RSA, AES, and one other. What is the full name of this fourth scheme?"
    ]

    for question in questions:
        log.info(f"Generating evaluation data for question: '{question}'")
        agent_response_with_acg, verified_contexts = await grounding_agent.generate_and_verify(question)
        agent_answer = re.sub(r"--- ACG_START ---.*?--- ACG_END ---", "", agent_response_with_acg, flags=re.DOTALL).strip()
        
        if agent_answer.startswith("❌ Agent unable to retrieve"):
            log.warning(f"Agent returned an error message for question '{question}' due to no grounding context. Skipping this question.")
            continue

        # For now, the reference answer is the same as the agent's answer,
        # but in a real scenario, this would be a human-written reference.
        reference_answer = agent_answer 

        evaluation_data.append({
            "user_input": question,
            "retrieved_contexts": verified_contexts,
            "response": agent_answer,
            "reference": reference_answer,
        })
    
    try:
        return evaluation_data
    finally:
        db_client.close()

if __name__ == "__main__":
    async def main():
        data = await get_evaluation_data()
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
