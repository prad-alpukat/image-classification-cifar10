import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# get data dari cifar10 dataset dari keras (source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data)
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

# class name dari dataset cifar10
class_name = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# menampilkan 16 gambar pertama dari dataset
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_name[training_labels[i][0]])
plt.show()

# (optional) membatasi jumlah training data untuk mempercepat proses training
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# jika ada model.keras yang sudah ada, maka load model tersebut
try:
    # load model
    model = models.load_model('model.keras')
except:
    # membuat model
    model = models.Sequential()

    # menambahkan layer conv untuk mengenali fitur dari gambar, relu untuk menghindari vanishing gradient, dan max pooling untuk mengurangi dimensi
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # menambahkan layer flatten untuk mengubah hasil dari layer conv menjadi 1 dimensi
    model.add(layers.Flatten())

    # menambahkan layer dense untuk menghubungkan hasil dari layer conv ke output layer
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # epoch = 10, karena hasil training dan testing sudah cukup baik
    history = model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

    # menampilkan hasil training
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

# save model
model.save('model.keras')

# menampilkan hasil testing
test_loss, test_acc = model.evaluate(testing_images, testing_labels, verbose=2)
print(test_acc)

# membuat prediksi
predictions = model.predict(testing_images)
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(testing_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = testing_labels[i][0]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel('{} ({})'.format(class_name[predicted_label], class_name[true_label]), color=color)
plt.show()



