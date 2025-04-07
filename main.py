from perceptron import PerceptronBipolar

# Definindo um conjunto de dados simples
# Cada sublista representa uma amostra com 4 características
# Os rótulos correspondentes são -1 ou 1
dataset = [
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
    [-1, -1, 1, 1],
    [1, 1, 1, -1],
    [-1, -1, -1, 1]
]

# Rótulos correspondentes para cada amostra
labels = [1, -1, -1, 1, 1, -1]

# Dividindo o dataset em treino e teste (2/3 para treino e 1/3 para teste)
train_size = int(len(dataset) * 2 / 3)
X_train, y_train = dataset[:train_size], labels[:train_size]
X_test, y_test = dataset[train_size:], labels[train_size:]

# Inicializando o Perceptron
perceptron = PerceptronBipolar(input_size=4, learning_rate=0.1)

# Treinando o Perceptron
perceptron.train(X_train, y_train, max_epochs=100)

# Testando o modelo
correct_predictions = 0
for inputs, label in zip(X_test, y_test):
    prediction = perceptron.predict(inputs)
    if prediction == label:
        correct_predictions += 1

# Calculando e exibindo a acurácia
accuracy = correct_predictions / len(y_test)
print(f"Acurácia no conjunto de teste: {
