import pandas as pd
import requests
import re
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
import time

class DatasetService:
    # Função para criar o dataset a partir do arquivo CSV
    def puxar_dados(self):
        caminho_do_csv = 'data.csv'
        nome_do_arquivo_de_saida = 'resultados_artigos.csv'
        DatasetService.extrair_discussoes_conclusoes(caminho_do_csv, nome_do_arquivo_de_saida)

    def buscar_dados(DatasetService, string_busca: str, numero: int):
        # Implementação da lógica para buscar dados com base na string de busca e no número
        pass

    @staticmethod
    def find_section_text(soup, section_title):
        """
        Encontra o título de uma seção e extrai todo o texto dos parágrafos
        seguintes até encontrar o próximo título.
        """
        # 1. Encontrar o cabeçalho da seção
        section_header = soup.find(['h2', 'h3'], string=lambda t: t and section_title.lower() in t.lower())
        
        # 2. Verificar se o cabeçalho foi encontrado
        if not section_header:
            return f"A seção '{section_title}' não foi encontrada."
        
        # 3. Coletar o conteúdo da seção
        content = []
        for sibling in section_header.find_next_siblings():
            # Para quando encontrar o próximo cabeçalho
            if sibling.name in ['h2', 'h3']:
                break
            # Adiciona o texto do parágrafo se for um elemento <p>
            if sibling.name == 'p':
                content.append(sibling.get_text(strip=True))
                
        # 4. Verificar se algum conteúdo foi coletado
        if not content:
            return f"A seção '{section_title}' foi encontrada, mas não continha texto de parágrafo extraível."
            
        # 5. Retornar o texto formatado
        return "\n\n".join(content)

    @staticmethod
    def extrair_discussoes_conclusoes(caminho_csv: str, arquivo_saida: str):
        """
        Lê um arquivo CSV com títulos e links de artigos, extrai as seções de
        "Discussão" e "Conclusão" de cada link e salva o resultado em um arquivo CSV.

        Args:
            caminho_csv (str): O caminho para o arquivo de entrada .csv.
            arquivo_saida (str): O nome do arquivo .csv onde os resultados serão salvos.
        """
        try:
            df = pd.read_csv(caminho_csv)
        except FileNotFoundError:
            print(f"Erro: O arquivo '{caminho_csv}' não foi encontrado.")
            return

        # Usar um 'user-agent' ajuda a simular um navegador real
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        resultados = []

        for index, row in df.iterrows():
            title = row['Title']
            url = row['Link']
            artigo = {
                "Title": title,
                "Link": url,
                "Discussao": "",
                "Conclusao": "",
                "Erro": ""
            }
            try:
                # Faz a requisição HTTP para obter o conteúdo da página
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()  # Lança um erro para status ruins (4xx ou 5xx)
                
                # Analisa o HTML da página
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extrai e salva a seção "Discussão"
                artigo["Discussao"] = DatasetService.find_section_text(soup, 'Discussion')
                # Extrai e salva a seção "Conclusão"
                artigo["Conclusao"] = DatasetService.find_section_text(soup, 'Conclusion')
                print(f"Processado com sucesso: {title}")
            except requests.exceptions.RequestException as e:
                artigo["Erro"] = f"Não foi possível acessar o artigo. Erro: {e}"
                print(artigo["Erro"])
            resultados.append(artigo)
            # Adiciona uma pausa para evitar sobrecarregar o servidor
            time.sleep(1)

        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(arquivo_saida, index=False, encoding='utf-8')
        print(f"\nProcesso concluído. Resultados salvos em '{arquivo_saida}'.")