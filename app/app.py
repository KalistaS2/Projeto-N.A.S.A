# 1. Importa a classe Flask e a função render_template
from flask import Flask, render_template

# 2. Cria uma instância da aplicação Flask
# __name__ é uma variável especial em Python que aponta para o nome do módulo atual
app = Flask(__name__)

# 3. Define uma rota (URL) usando um decorador
# O decorador @app.route('/') diz ao Flask que a função 'home' deve ser executada
# quando alguém acessar a página inicial do site
@app.route('/')
def home():
    # 4. Renderiza o arquivo HTML localizado na pasta 'templates'
    return render_template('index.html')

# 5. Ponto de entrada para executar a aplicação
# Este bloco só será executado se o script for chamado diretamente
if __name__ == '__main__':
    # Inicia o servidor de desenvolvimento do Flask
    # debug=True permite que o servidor reinicie automaticamente após alterações no código
    app.run(debug=True)