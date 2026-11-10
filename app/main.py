import logging

from fastapi import FastAPI

from .routes import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Meu RAG Engine",
    description="Oráculo de Editais e Artigos: busca precisa em PDFs acadêmicos.",
    version="1.0.0",
)

app.include_router(router)
