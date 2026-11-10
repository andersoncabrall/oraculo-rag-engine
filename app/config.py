import os

from dotenv import load_dotenv

load_dotenv() 

if not os.getenv("OPENAI_API_KEY"):
    # Falhar cedo e com mensagem clara é melhor que um erro obscuro depois.
    raise RuntimeError("Variável OPENAI_API_KEY não definida. Crie o arquivo .env.")

CHROMA_DIR = "./chroma_db"                 
COLLECTION_NAME = "vlab_docs"              
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000                          
CHUNK_OVERLAP = 200                        
TOP_K = 4                                  
EMBED_BATCH = 100                          

SYSTEM_PROMPT = (
    "Você é um assistente acadêmico. Responda à pergunta do usuário "
    "usando APENAS o contexto fornecido. Se a resposta não estiver no contexto, "
    "diga 'A informação não consta no documento'."
)
