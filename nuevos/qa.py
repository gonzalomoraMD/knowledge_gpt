
from typing import List
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from knowledge_gpt.core.prompts import STUFF_PROMPT
from langchain.docstore.document import Document
from knowledge_gpt.core.embedding import FolderIndex
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel

# Importar Chroma
import chromadb

class AnswerWithSources(BaseModel):
    answer: str
    sources: List[Document]

def query_folder(query: str, folder_index: FolderIndex, llm: BaseChatModel, use_chroma=False) -> AnswerWithSources:
    """Realiza una consulta en el índice de la carpeta utilizando los embeddings almacenados en Chroma si está habilitado."""
    if use_chroma:
        client = chromadb.Client()
        collection = client.get_collection("document_embeddings")
        query_embedding = OpenAIEmbeddings().embed([query])
        results = collection.query(query_embedding, top_k=10)
        docs = [Document(page_content=res["document"]["content"]) for res in results]
    else:
        # Aquí estaría el flujo estándar si no se usa Chroma
        docs = []

    answer_with_sources = llm.generate(docs, prompt=STUFF_PROMPT.format(question=query))
    return AnswerWithSources(answer=answer_with_sources["answer"], sources=docs)
