from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import LangChainTracer
from ..config.settings import settings


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.1,
        callbacks=[LangChainTracer(project_name=settings.langchain_project)],
    )
