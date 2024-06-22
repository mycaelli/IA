from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

class CNN:
    def __init__(self, image_width, image_height, batch_size, classes):
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.classes = classes

    def pre_processing(self, path):
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            directory=path,
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
            Dense(1, activation='sigmoid')  # Ativação sigmoid para classificação binária
        ])
        # Compilação do modelo com função de perda binary_crossentropy
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, path):
        train_generator = self.pre_processing(path)
        model = self.model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
        steps_per_epoch = max(1, train_generator.samples // self.batch_size)
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=15,
            callbacks=[early_stopping]
        )

        # Avaliação do modelo e plotagem da matriz de confusão
        self.evaluate_model(model, train_generator)

        return history

    def evaluate_model(self, model, generator):
        generator.reset()
        # Previsão utilizando o modelo treinado
        predictions = model.predict(generator, steps=generator.samples // self.batch_size + 1)
        # Conversão das previsões para rótulos de classe binária
        y_pred = (predictions > 0.5).astype(int)
        y_true = generator.classes
        # Impressão do relatório de classificação
        print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))
        # Cálculo e plotagem da matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, ['Class 0', 'Class 1'])

    def plot_confusion_matrix(self, cm, classes):
        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('Classe verdadeira')
        plt.xlabel('Classe predita')
        plt.tight_layout()
        plt.savefig('confusion_matrix_binary.png')
        plt.show()

# Organiza os subdiretórios
base_dir = './content/'
target_dir = './binary_images'

# Criar diretórios para cada classe
class1_dir = os.path.join(target_dir, 'classe1')
class2_dir = os.path.join(target_dir, 'classe2')
os.makedirs(class1_dir, exist_ok=True)
os.makedirs(class2_dir, exist_ok=True)

# Específico para o dataset de caracteres
# Parâmetro de classificação -> 1 a 26
letter = 1
letter_nums = 26

# Move cada imagem para o diretório da classe correta
for image in os.listdir(base_dir):
    if image.endswith('.png'):
        index = int(image.split('.')[0])  # Pega o número referente a cada imagem
        class_num = (index % letter_nums) + 1
        if (index % letter_nums) == (letter - 1):  # Decide para qual classe a imagem deve ir
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
