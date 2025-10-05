from service.datasetService import DatasetService
from service.pesquisaService import PesquisaService
from service.scoreService import ScoreService
service = PesquisaService()
datasetService = DatasetService()
teste_score = ScoreService()

string_busca = service.extrair_string_busca("estudo que utiliza rede complexas para analisar confiabilidade nas eleições")
print(string_busca)



teste_score.main()
