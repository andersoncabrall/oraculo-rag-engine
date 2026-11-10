import hashlib
import logging
import os
import tempfile
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_BATCH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
)
from .schemas import Fonte

logger = logging.getLogger("meu-rag")


class ErroDeNegocio(Exception):

    def __init__(self, status_code: int, mensagem: str):
        super().__init__(mensagem)
        self.status_code = status_code
        self.mensagem = mensagem

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Contexto:\n{contexto}\n\nPergunta: {pergunta}"),
    ]
)

chain = prompt | llm | StrOutputParser()

def indexar_pdf(conteudo: bytes, nome: str) -> dict:
    """Lê, fatia, gera embeddings e salva um PDF no ChromaDB."""
    if not conteudo:
        raise ErroDeNegocio(400, "O arquivo enviado está vazio.")
    if not conteudo.startswith(b"%PDF"):
        raise ErroDeNegocio(400, "O arquivo não parece ser um PDF válido.")

    caminho_tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(conteudo)
            caminho_tmp = tmp.name

        paginas = PyPDFLoader(caminho_tmp).load()
        paginas = [p for p in paginas if p.page_content.strip()]

        if not paginas:
            raise ErroDeNegocio(
                422,
                "Nenhum texto extraído. O PDF pode ser escaneado (imagem) e exigir OCR.",
            )

        chunks = splitter.split_documents(paginas)

        doc_hash = hashlib.sha256(conteudo).hexdigest()[:12]
        ids = []
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = nome  
            chunk.metadata["page"] = int(chunk.metadata.get("page", 0)) + 1  
            ids.append(f"{doc_hash}_{i}")

        for inicio in range(0, len(chunks), EMBED_BATCH):
            vectorstore.add_documents(
                chunks[inicio : inicio + EMBED_BATCH],
                ids=ids[inicio : inicio + EMBED_BATCH],
            )

        logger.info("Arquivo '%s' indexado: %d chunks.", nome, len(chunks))
        return {
            "mensagem": "PDF processado e indexado com sucesso.",
            "arquivo": nome,
            "paginas": len(paginas),
            "chunks_processados": len(chunks),
        }
    finally:
        if caminho_tmp and os.path.exists(caminho_tmp):
            os.remove(caminho_tmp)


def responder(pergunta: str) -> Tuple[str, List[Fonte]]:
    docs = vectorstore.similarity_search(pergunta, k=TOP_K)

    if not docs:
        raise ErroDeNegocio(
            404, "Nenhum documento indexado ainda. Envie um PDF em /upload primeiro."
        )

    contexto = "\n\n---\n\n".join(
        f"[Fonte: {d.metadata.get('source', '?')} | pág. {d.metadata.get('page', '?')}]\n"
        f"{d.page_content}"
        for d in docs
    )

    resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta})

    fontes = [
        Fonte(
            arquivo=d.metadata.get("source", "desconhecido"),
            pagina=d.metadata.get("page", "?"),
            trecho=d.page_content,
        )
        for d in docs
    ]
    return resposta, fontes
