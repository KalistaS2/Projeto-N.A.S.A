// Aguarda o HTML carregar completamente antes de executar o script
document.addEventListener('DOMContentLoaded', () => {

    // Mock Data: Seus dados JSON
const mockData = {
  vertente: 2,
  resumoResultado:
    "Este artigo sistematiza a literatura sobre suscetibilidade humana a phishing, focando em fatores como variáveis pessoais, situacionais e de suporte. Propõe o Modelo de Suscetibilidade a Phishing (PSM) em três etapas temporais, identificando lacunas em pesquisas, como estresse. Avalia impacto prático e qualidade de evidências, oferecendo diretrizes para estudos futuros em design experimental e proteção cibernética.",
  artigos: [
    {
      titulo: "A importância da educação digital na prevenção de ataques de engenharia social",
      link: "https://example.com/artigo1",
    },
    {
      titulo: "Modelos de avaliação de vulnerabilidade humana em segurança cibernética",
      link: "https://example.com/artigo2",
    },
    {
      titulo: "Medidas preventivas contra vazamento de dados em dispositivos IoT",
      link: "https://example.com/artigo3",
    },
    {
      titulo: "Análise de comportamento do usuário em ataques de phishing",
      link: "https://example.com/artigo4",
    },
  ],
};

    // Função para renderizar o dashboard com os dados
    function renderDashboard(data) {
        const summaryCard = document.getElementById('summaryCard');
        const articlesContainer = document.getElementById('articlesContainer');

        // 1. Limpa o conteúdo existente (importante para atualizações)
        summaryCard.innerHTML = '';
        articlesContainer.innerHTML = '';

        // 2. Preenche o Card de Resumo
        summaryCard.innerHTML = `
            <span class="vertente-badge">Vertente #${data.vertente}</span>
            <h3>Revisão Bibliografica Prevista</h3>
            <p>${data.resumoResultado}</p>
        `;

        // 3. Preenche os Cards de Artigos
        if (data.artigos && data.artigos.length > 0) {
            data.artigos.forEach(article => {
                const articleCardHTML = `
                    <div class="card article-card">
                        <h4>${article.titulo}</h4>
                        <a href="${article.link}" class="article-link" target="_blank" rel="noopener noreferrer">
                            Ler Artigo
                        </a>
                    </div>
                `;
                articlesContainer.innerHTML += articleCardHTML;
            });
        } else {
            articlesContainer.innerHTML = '<p>Nenhum artigo encontrado.</p>';
        }
    }

    // 4. Chama a função para renderizar o dashboard com os dados mockados
    renderDashboard(mockData);

});