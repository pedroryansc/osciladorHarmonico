import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import numpy as np

def interpolacaoPolinominal(x, y):
    polinomio = lagrange(x, y)

    print("Polinômio interpolador encontrado:")
    print(polinomio)

    x_grafico = np.arange(start=0, stop=12.1, step=0.1)
    y_grafico = polinomio(x_grafico)

    print(polinomio(5))

    gerarGraficoValidacao(x, y)

    plt.plot(x_grafico, y_grafico, color="blue", label="Interpolação polinominal")

    plt.legend()
    plt.ylim(-0.5, 1.5)
    plt.show()

def gerarGraficoValidacao(x_treinamento, y_treinamento):
    dfValidacao = pd.read_csv("Dados_validacao.csv")

    plt.scatter(x_treinamento, y_treinamento, color="red", label="Pontos originais")
    plt.plot(dfValidacao["Tempo"], dfValidacao["Saida_y"], color="green", label="Função correta")

    plt.xlabel("Tempo")

# Execução dos métodos

# Leitura do dataset com os dados de treinamento
df = pd.read_csv("Dados_treinamento.csv")

interpolacaoPolinominal(df["Tempo"], df["Saida_y"])