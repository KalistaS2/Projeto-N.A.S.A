import os
import fitz
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import spacy
import networkx as nx
import logging
from datetime import datetime
import json
from typing import List, Tuple, Dict, Any
import torch

class ScoreService:
    def __init__(self, model_name: str = "allenai-specter", relevance_weight: float = 0.7, impact_weight: float = 0.3):
        # Configuração de paths
        CURRENT_FILE = os.path.abspath(__file__)
        CURRENT_DIR = os.path.dirname(CURRENT_FILE)
        PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
        
        self.PDF_DIR = os.path.join(PROJECT_ROOT, "backend", "pdfs")
        self.OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
        self.MODEL_NAME = model_name
        self.RELEVANCE_WEIGHT = relevance_weight
        self.IMPACT_WEIGHT = impact_weight
        
        # Criar pastas se não existirem
        os.makedirs(self.PDF_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # Configurar logging
        self.setup_logging()
        
        # Carregar modelos
        self.load_models()
        
        # Arquivos de output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.OUTPUT_CSV = os.path.join(self.OUTPUT_DIR, f"scores_{timestamp}.csv")
        self.EMBEDDINGS_FILE = os.path.join(self.OUTPUT_DIR, f"embeddings_{timestamp}.npy")
        self.METADATA_FILE = os.path.join(self.OUTPUT_DIR, f"metadata_{timestamp}.json")
        
        self.logger.info(f"ScoreService inicializado com modelo: {model_name}")
        self.logger.info(f"PDFs encontrados: {len(self.get_pdf_files())}")

    def setup_logging(self):
        """Configura o sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.OUTPUT_DIR, "scoring.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_models(self):
        """Carrega os modelos de NLP"""
        try:
            self.logger.info("Carregando modelo spaCy...")
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.error("Modelo spaCy 'en_core_web_sm' não encontrado.")
            raise

        try:
            self.logger.info(f"Carregando SentenceTransformer: {self.MODEL_NAME}")
            self.model = SentenceTransformer(self.MODEL_NAME)
            # Testar se GPU está disponível
            if torch.cuda.is_available():
                self.logger.info("GPU detectada - usando CUDA")
                self.model = self.model.to('cuda')
            else:
                self.logger.info("Usando CPU")
        except Exception as e:
            self.logger.error(f"Erro ao carregar SentenceTransformer: {e}")
            raise

    def get_pdf_files(self) -> List[str]:
        """Retorna lista de arquivos PDF"""
        return sorted([f for f in os.listdir(self.PDF_DIR) if f.endswith(".pdf")])

    def extract_text_from_pdf(self, path: str) -> str:
        """Extrai texto de PDF com tratamento de erro melhorado"""
        text = ""
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
            if not text.strip():
                self.logger.warning(f"PDF vazio ou sem texto extraível: {os.path.basename(path)}")
        except Exception as e:
            self.logger.error(f"Erro ao extrair texto de {path}: {e}")
        return text

    def clean_text(self, text: str, min_length: int = 50) -> str:
        """Limpa e pré-processa texto com validação de qualidade"""
        if len(text.strip()) < min_length:
            return ""
        
        try:
            doc = self.nlp(text.lower())
            # Filtro mais sofisticado
            tokens = [
                t.lemma_ for t in doc 
                if not t.is_stop 
                and t.is_alpha 
                and len(t.lemma_) > 2  # Remove palavras muito curtas
                and t.lemma_ not in ['et', 'al', 'fig', 'table']  # Remove termos comuns indesejados
            ]
            cleaned = " ".join(tokens)
            return cleaned if len(cleaned) > min_length else ""
        except Exception as e:
            self.logger.error(f"Erro na limpeza do texto: {e}")
            return ""

    def embed_text_batch(self, texts: List[str], batch_size: int = 32) -> List[torch.Tensor]:
        """Gera embeddings em lote para melhor performance"""
        self.logger.info(f"Gerando embeddings para {len(texts)} documentos...")
        
        # Filtrar textos vazios
        valid_texts = [text for text in texts if text.strip()]
        if len(valid_texts) != len(texts):
            self.logger.warning(f"{len(texts) - len(valid_texts)} documentos vazios foram filtrados")
        
        try:
            embeddings = self.model.encode(
                valid_texts, 
                convert_to_tensor=True, 
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True  # Melhora similaridade de cosseno
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Erro ao gerar embeddings: {e}")
            raise

    def calculate_similarity_matrix(self, embeddings: torch.Tensor, names: List[str]) -> pd.DataFrame:
        """Calcula matriz de similaridade de forma eficiente"""
        self.logger.info("Calculando matriz de similaridade...")
        
        # Usar GPU se disponível
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embeddings = embeddings.to(device)
        
        # Calcular similaridades em lote
        with torch.no_grad():
            sim_matrix = util.cos_sim(embeddings, embeddings)
        
        # Converter para numpy e criar DataFrame
        sim_np = sim_matrix.cpu().numpy()
        return pd.DataFrame(sim_np, index=names, columns=names)

    def create_citation_network(self, names: List[str], similarity_df: pd.DataFrame, 
                              threshold: float = 0.3) -> nx.DiGraph:
        """Cria rede de citações baseada na similaridade"""
        G = nx.DiGraph()
        
        # Adicionar nós
        for name in names:
            G.add_node(name)
        
        # Adicionar arestas baseadas na similaridade
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i != j and similarity_df.iloc[i, j] > threshold:
                    # Direção baseada na similaridade (assimétrica)
                    if similarity_df.iloc[i, j] > similarity_df.iloc[j, i]:
                        G.add_edge(name1, name2, weight=similarity_df.iloc[i, j])
                    else:
                        G.add_edge(name2, name1, weight=similarity_df.iloc[j, i])
        
        self.logger.info(f"Rede criada com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas")
        return G

    def calculate_impact_scores(self, G: nx.DiGraph) -> Dict[str, float]:
        """Calcula scores de impacto usando PageRank"""
        try:
            pagerank_scores = nx.pagerank(G, alpha=0.85, weight='weight')
            # Normalizar scores entre 0 e 1
            if pagerank_scores:
                max_score = max(pagerank_scores.values())
                if max_score > 0:
                    pagerank_scores = {k: v/max_score for k, v in pagerank_scores.items()}
            return pagerank_scores
        except Exception as e:
            self.logger.error(f"Erro no cálculo do PageRank: {e}")
            return {node: 0.0 for node in G.nodes()}

    def save_results(self, df: pd.DataFrame, embeddings: torch.Tensor, metadata: Dict[str, Any]):
        """Salva resultados em múltiplos formatos"""
        # Salvar CSV principal
        df.to_csv(self.OUTPUT_CSV, index=False)
        self.logger.info(f"📊 Scores salvos em: {self.OUTPUT_CSV}")
        
        # Salvar embeddings (opcional - para análise futura)
        try:
            np.save(self.EMBEDDINGS_FILE, embeddings.cpu().numpy())
            self.logger.info(f"💾 Embeddings salvos em: {self.EMBEDDINGS_FILE}")
        except Exception as e:
            self.logger.warning(f" Não foi possível salvar embeddings: {e}")
        
        # Salvar metadados
        with open(self.METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.logger.info(f"📁 Metadados salvos em: {self.METADATA_FILE}")

    def analyze_results(self, df: pd.DataFrame):
        """Analisa e exibe estatísticas dos resultados"""
        self.logger.info("\n" + "="*50)
        self.logger.info("📈 ANÁLISE DOS RESULTADOS")
        self.logger.info("="*50)
        
        self.logger.info(f"📊 Total de artigos processados: {len(df)}")
        self.logger.info(f"🎯 Score médio: {df['final_score'].mean():.4f}")
        self.logger.info(f"📈 Score máximo: {df['final_score'].max():.4f}")
        self.logger.info(f"📉 Score mínimo: {df['final_score'].min():.4f}")
        self.logger.info(f"⭐ Top 5 artigos:")
        
        top_5 = df.nlargest(5, 'final_score')[['article', 'final_score']]
        for idx, row in top_5.iterrows():
            self.logger.info(f"   {row['article']}: {row['final_score']:.4f}")

    def main(self) -> pd.DataFrame:
        """Pipeline principal de processamento"""
        self.logger.info("🚀 Iniciando pipeline de scoring...")
        
        # Verificar arquivos PDF
        pdf_files = self.get_pdf_files()
        if not pdf_files:
            self.logger.error(f"❌ Nenhum arquivo PDF encontrado em: {self.PDF_DIR}")
            return pd.DataFrame()
        
        self.logger.info(f"📄 Processando {len(pdf_files)} arquivos PDF...")
        
        # Fase 1: Extração e limpeza
        texts, names, valid_indices = [], [], []
        for i, pdf_file in enumerate(tqdm(pdf_files, desc="Extraindo texto")):
            path = os.path.join(self.PDF_DIR, pdf_file)
            raw_text = self.extract_text_from_pdf(path)
            cleaned_text = self.clean_text(raw_text)
            
            if cleaned_text:
                texts.append(cleaned_text)
                names.append(pdf_file)
                valid_indices.append(i)
            else:
                self.logger.warning(f"⚠️  PDF ignorado (sem texto suficiente): {pdf_file}")
        
        if not texts:
            self.logger.error("❌ Nenhum texto válido pôde ser extraído dos PDFs")
            return pd.DataFrame()
        
        self.logger.info(f"✅ {len(texts)} documentos válidos para processamento")
        
        # Fase 2: Embeddings e similaridade
        embeddings = self.embed_text_batch(texts)
        similarity_df = self.calculate_similarity_matrix(embeddings, names)
        
        # Fase 3: Rede de impacto
        citation_network = self.create_citation_network(names, similarity_df)
        impact_scores = self.calculate_impact_scores(citation_network)
        
        # Fase 4: Cálculo de scores finais
        final_scores = []
        for i, name in enumerate(names):
            relevance = float(similarity_df[name].mean())
            impact_score = impact_scores.get(name, 0.0)
            final_score = (self.RELEVANCE_WEIGHT * relevance + 
                         self.IMPACT_WEIGHT * impact_score)
            
            final_scores.append({
                "article": name,
                "relevance": round(relevance, 6),
                "impact": round(impact_score, 6),
                "final_score": round(final_score, 6)
            })
        
        # Criar DataFrame final
        results_df = pd.DataFrame(final_scores)
        results_df = results_df.sort_values("final_score", ascending=False)
        
        # Salvar resultados
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_used": self.MODEL_NAME,
            "total_pdfs": len(pdf_files),
            "processed_pdfs": len(texts),
            "relevance_weight": self.RELEVANCE_WEIGHT,
            "impact_weight": self.IMPACT_WEIGHT,
            "parameters": {
                "similarity_threshold": 0.3,
                "pagerank_alpha": 0.85
            }
        }
        
        self.save_results(results_df, embeddings, metadata)
        self.analyze_results(results_df)
        
        # Exibir top 10
        self.logger.info("\n🏆 TOP 10 ARTIGOS:")
        self.logger.info(results_df.head(10).to_string(index=False))
        
        return results_df

# Para uso como módulo
def run_scoring_service(model_name: str = "allenai-specter", 
                       relevance_weight: float = 0.7, 
                       impact_weight: float = 0.3) -> pd.DataFrame:
    """
    Função conveniente para executar o serviço de scoring
    
    Args:
        model_name: Nome do modelo SentenceTransformer
        relevance_weight: Peso para relevância (0-1)
        impact_weight: Peso para impacto (0-1)
    
    Returns:
        DataFrame com os resultados
    """
    service = ScoreService(
        model_name=model_name,
        relevance_weight=relevance_weight,
        impact_weight=impact_weight
    )
    return service.main()

if __name__ == "__main__":
    # Executar com configurações padrão
    results = run_scoring_service()