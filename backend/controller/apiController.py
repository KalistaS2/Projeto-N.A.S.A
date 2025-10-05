# Importações de bibliotecas necessárias
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Importação do serviço de pesquisa
from backend.service.pesquisaService import PesquisaService
class apiController:
    router = APIRouter()

    class PesquisaRequest(BaseModel):
        texto: str
        numero: int

    @router.post("/pesquisar")
    async def predict(body: PesquisaRequest):
        # Inicializa o serviço de pesquisa
        service = PesquisaService()
        
        string_busca = service.extrair_string_busca(body.texto)
        # Acesse os dados com body.texto e body.numero
        
        datasetService = DatasetService()
        
        return {}