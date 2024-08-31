
from langchain.vectorstores import VectorStore
from knowledge_gpt.core.parsing import File
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from typing import List, Type
from langchain.docstore.document import Document
from knowledge_gpt.core.debug import FakeVectorStore, FakeEmbeddings

# Importar Chroma
import chromadb

class FolderIndex:
    """Index for a collection of files (a folder)"""

    def __init__(self, files: List[File], use_chroma=False):
        self.files = files
        self.use_chroma = use_chroma
        if self.use_chroma:
            self.client = chromadb.Client()
            self.collection = self.client.create_collection("document_embeddings")

    def add_embeddings_to_chroma(self, embeddings, metadata):
        if self.use_chroma:
            self.collection.add(embeddings, metadata=metadata)

def embed_files(files: List[File], use_chroma=False) -> FolderIndex:
    """Genera embeddings para los archivos y los almacena en Chroma si está habilitado."""
    folder_index = FolderIndex(files, use_chroma)
    for file in files:
        for doc in file.docs:
            embedding = OpenAIEmbeddings().embed([doc.page_content])
            if use_chroma:
                folder_index.add_embeddings_to_chroma(embedding, {"document_id": file.id})
    return folder_index
