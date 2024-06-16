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
    class_mode='categorical',  # Não há classes a dividir
    color_mode='grayscale'
    )

    print("Número de amostras carregadas:", train_generator.samples)

    return train_generator

  def model(self):
    model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_width, self.image_height, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'), # 26 classes
    Dense(self.classes, activation='softmax')  # Ajuste para classificação multiclasse
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model
  
  def plot_history(self, history):
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
    self.plot_history(history)

# organiza os subdiretorios

base_dir = './content/'
target_dir = './multiclass_images'
classes = 26

# cria o diretorio para cada classe (A...Z)
for i in range (1, classes + 1):
  class_dir = os.path.join(target_dir, f'classe{i}')
  os.makedirs(class_dir, exist_ok=True)

# move cada imagem para o diretorio da classe correta
for image in os.listdir(base_dir):
  if image.endswith('.png'):
    index = int(image.split('.')[0]) # pega o numero referente a cada imagem
    class_num = (index % classes) + 1
    src_path = os.path.join(base_dir, image)
    dest_path = os.path.join(target_dir, f'classe{class_num}', image)
    shutil.copy(src_path, dest_path)


cnn = CNN(image_width=12, image_height=10, batch_size=26, classes=classes)
cnn.train(path=target_dir)
