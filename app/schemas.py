"""Schemas Pydantic: validação automática de entrada e saída da API."""

from typing import List

from pydantic import BaseModel, Field


class PerguntaRequest(BaseModel):
    pergunta: str = Field(..., min_length=3, max_length=1000)


class Fonte(BaseModel):
    arquivo: str
    pagina: int | str
    trecho: str


class RespostaResponse(BaseModel):
    resposta: str
    fontes: List[Fonte]
