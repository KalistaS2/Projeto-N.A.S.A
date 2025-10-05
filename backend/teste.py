from service.datasetService import DatasetService
from service.pesquisaService import PesquisaService
<<<<<<< HEAD
from service.scoreService import ScoreService
=======

>>>>>>> be5957da33b6e591043ec0b33f27fe8e1f1c2d83
service = PesquisaService()
datasetService = DatasetService()

string_busca = service.extrair_string_busca("estudo que utiliza rede complexas para analisar confiabilidade nas eleições")
print(string_busca)

<<<<<<< HEAD

teste_score = ScoreService()

teste_score.main()
=======
datasetService.puxar_dados()
>>>>>>> be5957da33b6e591043ec0b33f27fe8e1f1c2d83
