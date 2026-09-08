import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def interpolacaoPolinominal(x, y):
    polinomio = lagrange(x, y)

    print("Polinômio interpolador encontrado:")
    print(polinomio)

    x_grafico = np.arange(start=0, stop=12.1, step=0.1)
    y_grafico = polinomio(x_grafico)

    gerarGraficoValidacao(x, y)

    plt.plot(x_grafico, y_grafico, color="blue", label="Interpolação polinominal")

    plt.title("Interpolação Polinominal - Polinômio de grau 8")
    plt.legend()
    plt.show()

def minimosQuadrados(x, y):
    # Cálculo dos coeficientes do polinômio de grau 4 a partir do método dos mínimos quadrados
    a, b, c, d, e = np.polyfit(x, y, 4)

    x_grafico = np.arange(start=0, stop=12.1, step=0.1)
    y_grafico = a * (x_grafico ** 4) + b * (x_grafico ** 3) + c * (x_grafico ** 2) + d * x_grafico + e

    gerarGraficoValidacao(x, y)

    plt.plot(x_grafico, y_grafico, color="blue", label="Mínimos quadrados")

    plt.title("Mínimos Quadrados - Polinômio de grau 4")
    plt.legend()
    plt.show()

def redesNeurais(x, y):
    mlp = Sequential()
    mlp.add(Dense(1, input_shape=(1,), activation="tanh"))
    mlp.add(Dense(1))

    print(mlp.summary())

def gerarGraficoValidacao(x_treinamento, y_treinamento):
    dfValidacao = pd.read_csv("Dados_validacao.csv")

    plt.scatter(x_treinamento, y_treinamento, color="red", label="Pontos originais")
    plt.plot(dfValidacao["Tempo"], dfValidacao["Saida_y"], color="green", label="Função correta")

    plt.xlabel("Tempo")
    plt.ylim(-0.5, 1.5)

# Execução dos métodos

# Leitura do dataset com os dados de treinamento
df = pd.read_csv("Dados_treinamento.csv")

# interpolacaoPolinominal(df["Tempo"], df["Saida_y"])
# minimosQuadrados(df["Tempo"], df["Saida_y"])
redesNeurais(df["Tempo"], df["Saida_y"])