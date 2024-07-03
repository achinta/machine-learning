from enum import Enum
import os
from fastembed import TextEmbedding

embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")

class LLMInferenceProvider(Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    GROQ = "groq"

def get_llm(llm_inference_provider: LLMInferenceProvider):
    if llm_inference_provider == LLMInferenceProvider.OPENAI:
        from langchain_openai import ChatOpenAI 
        return ChatOpenAI()
    elif llm_inference_provider == LLMInferenceProvider.AZURE_OPENAI:
        from langchain_openai import AzureChatOpenAI
        # return AzureChatOpenAI(api_version="202"
        return AzureChatOpenAI(model="GPT4", temperature=1, api_version='2024-02-01')
    elif llm_inference_provider == LLMInferenceProvider.GROQ:
        from langchain_groq import GroqChat
        return GroqChat()
    else:
        raise ValueError(f"LLM inference provider {llm_inference_provider} not supported.")
    


