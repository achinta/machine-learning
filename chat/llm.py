from enum import Enum

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
        return AzureChatOpenAI(model="deepiq35turbo", temperature=1, api_version='2024-02-01')
    elif llm_inference_provider == LLMInferenceProvider.GROQ:
        from langchain_groq import GroqChat
        return GroqChat()
    else:
        raise ValueError(f"LLM inference provider {llm_inference_provider} not supported.")

