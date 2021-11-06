import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('personagens.csv')

X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values

y = (y == 'Bart')

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
    X, y, test_size=0.2)

# (entradas + saidas) /2
# 6 -> 4 -> 4 ->4 ->1

rede_neural = tf.keras.models.Sequential()
rede_neural.add(tf.keras.layers.Dense(
    units=4, activation='relu', input_shape=(6,)))
rede_neural.add(tf.keras.layers.Dense(units=4, activation='relu'))
rede_neural.add(tf.keras.layers.Dense(units=4, activation='relu'))
rede_neural.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

rede_neural.compile(
    optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
historico = rede_neural.fit(
    X_treinamento, y_treinamento, epochs=50, validation_split=0.1)

previsoes = rede_neural.predict(X_teste)
previsoes = (previsoes > 0.5)

accuracy_score(previsoes, y_teste)
