import tensorflow as tf

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

dy_dx = tape.gradient(y, x)

print(f"\nValor de y (x^2): {y.numpy():.2f}")
print(f"O gradiente (derivada) de y = x^2 no ponto x = 3.0 é igual a {dy_dx.numpy():.2f}")