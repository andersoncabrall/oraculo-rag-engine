# Oráculo de Editais e Artigos

Sistema RAG que responde perguntas sobre PDFs como editais 
resoluções, artigos, usando apenas o conteúdo dos documentos enviados.

**Stack:** Python 3.10+, FastAPI, LangChain, ChromaDB, OpenAI, Docker.

## Arquitetura

```
app/
├── main.py      # cria o FastAPI e registra as rotas
├── routes.py    # endpoints HTTP (/upload e /perguntar)
├── rag.py       # núcleo: ingestão e consulta
├── schemas.py   # modelos Pydantic
└── config.py    # variáveis de ambiente e constantes
```

**Ingestão:** PDF → PyPDFLoader → RecursiveCharacterTextSplitter → embeddings → ChromaDB
**Consulta:** pergunta → embedding → top-4 no ChromaDB → prompt restrito → gpt-4o-mini → resposta + fontes

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
cp .env.example .env             
uvicorn app.main:app --reload
```

Docs interativas: http://localhost:8000/docs

### Docker

```bash
docker build -t pdf-rag-engine .
docker run -p 8000:8000 --env-file .env -v $(pwd)/chroma_db:/app/chroma_db pdf-rag-engine
```

## Uso

```bash
curl -X POST http://localhost:8000/upload -F "file=@edital.pdf"

curl -X POST http://localhost:8000/perguntar \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Qual o valor da bolsa?"}'
```

## Limitações e próximos passos

- PDFs escaneados exigem OCR
- Tabelas são extraídas de forma imperfeita
- Busca apenas semântica (evoluir para híbrida + reranking)
- Sem avaliação formal (evoluir com RAGAS) e sem autenticação
