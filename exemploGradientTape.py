# Importando a biblioteca
import tensorflow as tf

# Definindo o tensor (vetor x)
x = tf.constant([1.0, 2.0, 3.0])

# Abrindo o bloco do GradientTape e gravando a função y
with tf.GradientTape() as tape:
    tape.watch(x)
    y = (x ** 2) + 3 * x # y = x² + 3x

# Calculando o gradiente (derivada) de y e aplicando os valores do vetor x
gradiente = tape.gradient(y, x) # dy/dx = 2x + 3

print(gradiente)