from service.pesquisaService import PesquisaService
from service.scoreService import ScoreService
service = PesquisaService()

string_busca = service.extrair_string_busca("estudo que utiliza rede complexas para analisar confiabilidade nas eleições")
print(string_busca)


teste_score = ScoreService()

teste_score.main()