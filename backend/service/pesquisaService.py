import spacy
from deep_translator import GoogleTranslator

# Use o modelo em português
nlp = spacy.load("pt_core_news_sm")

class PesquisaService:
    
    @staticmethod
    def extrair_string_busca(texto: str) -> str:
        
        doc = nlp(texto)
        palavras_chave = sorted(set([token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and token.is_alpha]))
        palavras_chave_traduzidas = PesquisaService.traduzir_lista_palavras(palavras_chave)
        palavras_filtradas = [p for p in palavras_chave_traduzidas if p]
        string_de_busca = " AND ".join(palavras_filtradas)
        return string_de_busca
    
    def traduzir_lista_palavras(palavras: list[str]) -> list[str]:
        tradutor = GoogleTranslator(source='pt', target='en')
        palavras_traduzidas = []

        for palavra in palavras:
            try:
                traducao = tradutor.translate(palavra)
                palavras_traduzidas.append(traducao)
            except Exception as e:
                print(f"Erro ao traduzir '{palavra}': {e}")
                palavras_traduzidas.append(palavra)  # mantém a palavra original em caso de erro

        return palavras_traduzidas