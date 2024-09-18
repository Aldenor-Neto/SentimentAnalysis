import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import torch
import os

# Carregar o dataset
df = pd.read_csv('archive/chennai_reviews.csv')

# Excluir colunas não necessárias e renomear as colunas
df = df[['Review_Text', 'Sentiment']]

# Ajustar valores da coluna 'Sentiment'
df['Sentiment'] = df['Sentiment'].astype(str)
df['Sentiment'] = pd.to_numeric(df['Sentiment'], errors='coerce')
df['Sentiment'] = df['Sentiment'].fillna(df['Sentiment'].mode()[0]).astype(int) - 1

# Converter textos para strings
df['Review_Text'] = df['Review_Text'].astype(str)

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['Review_Text'], df['Sentiment'], test_size=0.2, random_state=42)

# Tokenização e padronização das sequências
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = max(len(seq) for seq in X_train_seq)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

# Modelo RNN
print("Iniciando o treinamento do Modelo RNN...")
model_rnn = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
    SimpleRNN(64),
    Dense(3, activation='softmax')
])

model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_rnn = model_rnn.fit(X_train_pad, y_train, epochs=10, validation_split=0.2)

# Modelo Transformer
print("Iniciando o treinamento do Modelo Transformer...")
inputs = Input(shape=(max_length,))
x = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(inputs)
x = GlobalAveragePooling1D()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(3, activation='softmax')(x)

model_transformer = Model(inputs, outputs)
model_transformer.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_transformer = model_transformer.fit(X_train_pad, y_train, epochs=10, validation_split=0.2)

# Diretório e nome da imagem
image_path = 'comparacao_modelos.png'

# Plotar a evolução da função custo
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_rnn.history['loss'], label='Treinamento RNN')
plt.plot(history_rnn.history['val_loss'], label='Validação RNN')
plt.title('Evolução da Função Custo - RNN')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_transformer.history['loss'], label='Treinamento Transformer')
plt.plot(history_transformer.history['val_loss'], label='Validação Transformer')
plt.title('Evolução da Função Custo - Transformer')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()

# Salvar a imagem, substituindo se já existir
if os.path.exists(image_path):
    os.remove(image_path)
plt.savefig(image_path)
print(f"Gráfico salvo em: {image_path}")

plt.show()

# Avaliar o modelo RNN
print("Avaliação final do Modelo RNN...")
loss_rnn, accuracy_rnn = model_rnn.evaluate(X_test_pad, y_test)
print(f'Taxa de Acerto RNN: {accuracy_rnn * 100:.2f}%')

# Avaliar o modelo Transformer
print("Avaliação final do Modelo Transformer...")
loss_transformer, accuracy_transformer = model_transformer.evaluate(X_test_pad, y_test)
print(f'Taxa de Acerto Transformer: {accuracy_transformer * 100:.2f}%')

# Comparação dos modelos
accuracies_rnn = []
accuracies_transformer = []

for i in range(5):
    # Recriar e treinar o modelo RNN
    model_rnn = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
        SimpleRNN(64),
        Dense(3, activation='softmax')
    ])
    model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_rnn.fit(X_train_pad, y_train, epochs=10, validation_split=0.2, verbose=0)

    # Recriar e treinar o modelo Transformer
    inputs = Input(shape=(max_length,))
    x = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)

    model_transformer = Model(inputs, outputs)
    model_transformer.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_transformer.fit(X_train_pad, y_train, epochs=10, validation_split=0.2, verbose=0)

    # Avaliar modelos
    loss_rnn, accuracy_rnn = model_rnn.evaluate(X_test_pad, y_test, verbose=0)
    loss_transformer, accuracy_transformer = model_transformer.evaluate(X_test_pad, y_test, verbose=0)

    accuracies_rnn.append(accuracy_rnn)
    accuracies_transformer.append(accuracy_transformer)

# Usar PyTorch para calcular a média e o desvio padrão das acurácias
accuracies_rnn_tensor = torch.tensor(accuracies_rnn)
accuracies_transformer_tensor = torch.tensor(accuracies_transformer)

media_rnn = torch.mean(accuracies_rnn_tensor).item()
desvio_padrao_rnn = torch.std(accuracies_rnn_tensor).item()

media_transformer = torch.mean(accuracies_transformer_tensor).item()
desvio_padrao_transformer = torch.std(accuracies_transformer_tensor).item()

print("\nEstatísticas de Desempenho:")
print(f'Média de acurácia RNN: {media_rnn * 100:.2f}%, Desvio-padrão: {desvio_padrao_rnn * 100:.2f}%')
print(
    f'Média de acurácia Transformer: {media_transformer * 100:.2f}%, Desvio-padrão: {desvio_padrao_transformer * 100:.2f}%')
