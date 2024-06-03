import numpy as np

import matplotlib.pyplot as plt

import os


def carregar_dados():
    caminho_entrada = 'X.npy'
    caminho_saida = 'Y_classe.npy'

    dados_entrada = np.load(caminho_entrada)
    dados_saida = np.load(caminho_saida)

    dados_entrada = dados_entrada.reshape(dados_entrada.shape[0], -1)

    return dados_entrada, dados_saida

class MLP:
    def __init__(self, tamanho_cam_entrada, tamanhos_cam_ocultas, tamanho_cam_saida, taxa_aprendizado=0.01):
        self.tamanhos_cam_ocultas = tamanhos_cam_ocultas
        self.pesos = []
        self.biases = []

        # Inicialização dos pesos e biases
        tamanho_cam_anterior = tamanho_cam_entrada + 1
        for tamanho_cam_oculta in tamanhos_cam_ocultas:
            self.pesos.append(np.random.uniform(-0.5, 0.5, [tamanho_cam_anterior, tamanho_cam_oculta]))
            self.biases.append(np.full(tamanho_cam_oculta, 1, dtype=float))
            tamanho_cam_anterior = tamanho_cam_oculta + 1

        self.pesos.append(np.random.uniform(-0.5, 0.5, [tamanho_cam_anterior, tamanho_cam_saida]))
        self.taxa_aprendizado = taxa_aprendizado

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        ativacoes = [np.insert(X, 0, 1, axis=1)]
        for i in range(len(self.tamanhos_cam_ocultas)):
            soma_ponderada = np.dot(ativacoes[-1], self.pesos[i])
            ativacao = self.sigmoid(soma_ponderada)
            ativacoes.append(np.insert(ativacao, 0, 1, axis=1))

        soma_ponderada_final = np.dot(ativacoes[-1], self.pesos[-1])
        saida = self.sigmoid(soma_ponderada_final)

        return saida, ativacoes

    def retropropagacao(self, X, Y, saida, ativacoes):
        erro_saida = Y - saida
        delta_saida = erro_saida * self.derivada_sigmoid(saida)

        deltas = [delta_saida]
        for i in reversed(range(len(self.tamanhos_cam_ocultas))):
            erro = np.dot(deltas[0], self.pesos[i + 1].T[:, 1:])
            delta = erro * self.derivada_sigmoid(ativacoes[i + 1][:, 1:])
            deltas.insert(0, delta)

        self.pesos[-1] += self.taxa_aprendizado * np.dot(ativacoes[-1].T, delta_saida)
        for i in range(len(self.tamanhos_cam_ocultas) - 1, -1, -1):
            self.pesos[i] += self.taxa_aprendizado * np.dot(ativacoes[i].T, deltas[i])

    def treinar(self, X_treinamento, Y_treinamento, X_validacao, Y_validacao, epocas=1000):
        acuracias = []
        erros = []
        val_acuracias = []
        val_erros = []

        for epoca in range(epocas):
            saida, ativacoes = self.feedforward(X_treinamento)
            self.retropropagacao(X_treinamento, Y_treinamento, saida, ativacoes)

            erro = np.mean(np.square(Y_treinamento - saida))
            erros.append(erro)

            acuracia = accuracy_score(Y_treinamento.argmax(axis=1), np.round(saida).argmax(axis=1))
            acuracias.append(acuracia)

            # Validação
            saida_validacao, _ = self.feedforward(X_validacao)
            erro_validacao = np.mean(np.square(Y_validacao - saida_validacao))
            val_erros.append(erro_validacao)
            acuracia_validacao = accuracy_score(Y_validacao.argmax(axis=1), np.round(saida_validacao).argmax(axis=1))
            val_acuracias.append(acuracia_validacao)

        return acuracias, erros, val_acuracias, val_erros

    def prever(self, X):
        saida, _ = self.feedforward(X)
        return saida

def busca_em_grade(X_treinamento, Y_treinamento, X_validacao, Y_validacao, grid_parametros):
    melhores_parametros = None
    melhor_acuracia = 0
    resultados = []

    for taxa_aprendizado in grid_parametros['taxa_aprendizado']:
        for tamanhos_cam_ocultas in grid_parametros['tamanhos_cam_ocultas']:
            mlp = MLP(tamanho_cam_entrada=X_treinamento.shape[1], tamanhos_cam_ocultas=tamanhos_cam_ocultas,
                      tamanho_cam_saida=Y_treinamento.shape[1], taxa_aprendizado=taxa_aprendizado)
            acuracias, erros, val_acuracias, val_erros = mlp.treinar(X_treinamento, Y_treinamento, X_validacao, Y_validacao, epocas=2000)
            media_val_acuracia = np.mean(val_acuracias)

            resultados.append({
                'taxa_aprendizado': taxa_aprendizado,
                'tamanhos_cam_ocultas': tamanhos_cam_ocultas,
                'val_acuracia': media_val_acuracia
            })

            if media_val_acuracia > melhor_acuracia:
                melhor_acuracia = media_val_acuracia
                melhores_parametros = {'taxa_aprendizado': taxa_aprendizado, 'tamanhos_cam_ocultas': tamanhos_cam_ocultas}

    return melhores_parametros, resultados

# Criar o diretório de saída, se não existir
diretorio_saida = 'C:\\Users\\gui02\\Downloads\\EP IA\\IA-main\\arquivos de saida'
if not os.path.exists(diretorio_saida):
    os.makedirs(diretorio_saida)

# Carregar os dados dos caracteres
X, Y = carregar_dados()

# Normalização dos dados
escalador = StandardScaler()
X = escalador.fit_transform(X)

# Manter os últimos 130 valores para teste
X_teste = X[-130:]
Y_teste = Y[-130:]
X_restante = X[:-130]
Y_restante = Y[:-130]

# Dividir os dados restantes em treino (60%) e validação (40%)
X_treinamento, X_validacao, Y_treinamento, Y_validacao = train_test_split(X_restante, Y_restante, test_size=0.4, random_state=42)

# Definir a grade de parâmetros para o Grid Search
grid_parametros = {
    'taxa_aprendizado': [0.001, 0.005, 0.01],
    'tamanhos_cam_ocultas': [[20, 10], [40, 20], [50, 30]]
}

# Executar o Grid Search
melhores_parametros, resultados = busca_em_grade(X_treinamento, Y_treinamento, X_validacao, Y_validacao, grid_parametros)

# Exibir os melhores parâmetros
print("Melhores parâmetros encontrados:", melhores_parametros)

# Treinar o MLP com os melhores parâmetros encontrados
mlp = MLP(tamanho_cam_entrada=120, tamanhos_cam_ocultas=melhores_parametros['tamanhos_cam_ocultas'], tamanho_cam_saida=26,
          taxa_aprendizado=melhores_parametros['taxa_aprendizado'])
pesos_iniciais = {'pesos_entrada_para_oculta': mlp.pesos[0], 'pesos_oculta_para_saida': mlp.pesos[-1]}
acuracias, erros, val_acuracias, val_erros = mlp.treinar(X_treinamento, Y_treinamento, X_validacao, Y_validacao, epocas=2000)
pesos_finais = {'pesos_entrada_para_oculta': mlp.pesos[0], 'pesos_oculta_para_saida': mlp.pesos[-1]}

print("Pesos Iniciais:")
print(pesos_iniciais)
print("\nPesos Finais:")
print(pesos_finais)

# Plotar as acurácias de treinamento e validação
plt.figure(figsize=(10, 6))
plt.plot(acuracias, label='Acurácia de Treinamento')
plt.plot(val_acuracias, label='Acurácia de Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.title('Acurácia ao longo das Épocas')
plt.legend()
plt.savefig(os.path.join(diretorio_saida, "grafico_acuracia.png"))
plt.show()

# Plotar os erros de treinamento e validação
plt.figure(figsize=(10, 6))
plt.plot(erros, label='Erro de Treinamento')
plt.plot(val_erros, label='Erro de Validação')
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.title('Erro de Treinamento e Validação ao longo das Épocas')
plt.legend()
plt.savefig(os.path.join(diretorio_saida, "grafico_erro.png"))
plt.show()

# Fazer previsões com o conjunto de teste
previsoes_teste = mlp.prever(X_teste)

# Salvar arquivos de saída
np.save(os.path.join(diretorio_saida, "pesos_iniciais.npy"), pesos_iniciais)
np.save(os.path.join(diretorio_saida, "pesos_finais.npy"), pesos_finais)
np.save(os.path.join(diretorio_saida, "erros_treinamento.npy"), erros)
np.save(os.path.join(diretorio_saida, "erros_validacao.npy"), val_erros)
np.save(os.path.join(diretorio_saida, "previsoes_teste.npy"), previsoes_teste)

# Salvar hiperparâmetros
hiperparametros = {
    "tamanho_cam_entrada": 120,
    "tamanhos_cam_ocultas": melhores_parametros['tamanhos_cam_ocultas'],
    "tamanho_cam_saida": 26,
    "taxa_aprendizado": melhores_parametros['taxa_aprendizado'],
    "epocas": 2000
}
with open(os.path.join(diretorio_saida, "hiperparametros.txt"), "w") as f:
    for chave, valor in hiperparametros.items():
        f.write(f"{chave}: {valor}\n")