from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

import os
import shutil

class CNN:
  def __init__ (self, image_width, image_height, batch_size, classes):
    self.image_width = image_width
    self.image_height = image_height
    self.batch_size = batch_size
    self.classes = classes

  
  def pre_processing(self, path):
    # normalizacao dos pixels das imagens para valores entte 0 e 1
    train_datagen = ImageDataGenerator(
        rescale=1./255  # Apenas reescala, sem transformações geométricas
    )

    train_generator = train_datagen.flow_from_directory(
    directory=path,  # Caminho para o subdiretório contendo todas as imagens
    target_size=(self.image_width,  self.image_height),
    batch_size=self.batch_size,
    class_mode='binary',
    color_mode='grayscale'
    )

    print("Número de amostras carregadas:", train_generator.samples)

    return train_generator

  def model(self):
    model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_width, self.image_height, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(self.classes, activation='sigmoid')  # Ajuste para classificação multiclasse
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

  def train(self, path):
    train_generator = self.pre_processing(path)
    model = self.model()

    # Early stopping para evitar overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1)

    # define que todas as imagens serao usadas em cada epoca
    steps_per_epoch = max(1, train_generator.samples // self.batch_size)

    history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=15,
    callbacks=[early_stopping]
    )

    return history

# organiza os subdiretorios

base_dir = './content/'
target_dir = './binary_images'

# criar diretórios para cada classe
class1_dir = os.path.join(target_dir, 'classe1')
class2_dir = os.path.join(target_dir, 'classe2')
os.makedirs(class1_dir, exist_ok=True)
os.makedirs(class2_dir, exist_ok=True)


## especifico para o dataset de caracteres
# parametro de classificacao -> 1 a 26
letter = 1
letter_nums = 26

# move cada imagem para o diretorio da classe correta
for image in os.listdir(base_dir):
  if image.endswith('.png'):
    index = int(image.split('.')[0]) # pega o numero referente a cada imagem
    class_num = (index % letter_nums) + 1
    if (index % letter_nums) == (letter - 1): # decide para qual classe a imagem deve ir
      src_path = os.path.join(base_dir, image)
      dest_path = os.path.join(class1_dir,  image)
      shutil.copy(src_path, dest_path)
    else:
      src_path = os.path.join(base_dir, image)
      dest_path = os.path.join(class2_dir,  image)
      shutil.copy(src_path, dest_path)


cnn = CNN(image_width=12, image_height=10, batch_size=32, classes=1)
history = cnn.train(path=target_dir)


acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
# Plotar acurácia
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Acurácia de Treinamento')
plt.title('Acurácia de Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
# Plotar perda
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Perda de Treinamento')
plt.title('Perda de Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()