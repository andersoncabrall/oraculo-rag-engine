import logging

import openai
from fastapi import APIRouter, File, HTTPException, UploadFile

from . import rag
from .rag import ErroDeNegocio
from .schemas import PerguntaRequest, RespostaResponse

logger = logging.getLogger("meu-rag")
router = APIRouter()


def _traduzir_erro_openai(e: Exception) -> HTTPException:
    if isinstance(e, openai.AuthenticationError):
        return HTTPException(502, "Chave da OpenAI inválida. Verifique o .env.")
    if isinstance(e, openai.RateLimitError):
        return HTTPException(429, "Limite/cota da OpenAI atingido. Tente mais tarde.")
    if isinstance(e, openai.APIConnectionError):
        return HTTPException(502, "Não foi possível conectar à API da OpenAI.")
    return HTTPException(500, "Erro interno ao processar a requisição.")


# OpenAI são bloqueantes; com `def` o FastAPI usa uma thread e não trava o servidor.
@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    nome = file.filename or "documento.pdf"

    if not nome.lower().endswith(".pdf"):
        raise HTTPException(400, "Envie um arquivo com extensão .pdf.")

    try:
        conteudo = file.file.read()
        return rag.indexar_pdf(conteudo, nome)
    except ErroDeNegocio as e:
        raise HTTPException(e.status_code, e.mensagem)
    except Exception as e:
        logger.exception("Falha ao processar upload de '%s'", nome)
        raise _traduzir_erro_openai(e)
    finally:
        file.file.close()


@router.post("/perguntar", response_model=RespostaResponse)
def perguntar(req: PerguntaRequest):
    try:
        resposta, fontes = rag.responder(req.pergunta)
        return RespostaResponse(resposta=resposta, fontes=fontes)
    except ErroDeNegocio as e:
        raise HTTPException(e.status_code, e.mensagem)
    except Exception as e:
        logger.exception("Falha ao responder pergunta")
        raise _traduzir_erro_openai(e)
