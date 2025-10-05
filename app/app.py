from flask import Flask, render_template, request, json

app = Flask(__name__)

def realizar_busca(query):
    try:
        with open('mockdata.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        if query.lower() in data['resumoResultado'].lower():
            return data
        artigos_filtrados = [artigo for artigo in data['artigos'] if query.lower() in artigo['titulo'].lower()]
        if artigos_filtrados:
            return {"resumoResultado": f"Exibindo {len(artigos_filtrados)} artigos para '{query}'", "artigos": artigos_filtrados}
    except FileNotFoundError:
        return None
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query', '')
    resultados = realizar_busca(query) or {"resumoResultado": "Nenhum resultado encontrado."}
    return render_template('dashboard.html', query=query, resultados=resultados)

if __name__ == '__main__':
    app.run(debug=True)