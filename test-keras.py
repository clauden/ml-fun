import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as pltfm
import sys


trn_imgs = trn_labels = tst_imgs = tst_labels = []
prds = []
epox = 5


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
 
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def visualize(idx):
	plt.figure(figsize=(6,3))

	plt.subplot(1, 2, 1)
	plot_image(idx, prds, tst_labels, tst_imgs)
	
	plt.subplot(1, 2, 2)
	plot_value_array(idx, prds, tst_labels)

	plt.show()

	# Plot the first X test images, their predicted label, and the true label
	# Color correct predictions in blue, incorrect predictions in red
	num_rows = 5
	num_cols = 3
	num_images = num_rows*num_cols

	plt.figure(figsize=(2*2*num_cols, 2*num_rows))

	for i in range(num_images):
	  plt.subplot(num_rows, 2*num_cols, 2*i+1)
	  plot_image(i, prds, tst_labels, tst_imgs)
	  plt.subplot(num_rows, 2*num_cols, 2*i+2)
	  plot_value_array(i, prds, tst_labels)

	plt.show()


### main begins ###


print(tf.__version__)
# pltfm.FontProperties(size='small')

if len(sys.argv) > 1:
	epox = int(sys.argv[1])
print("epox = ", epox)

dataset = keras.datasets.fashion_mnist

(trn_imgs, trn_labels), (tst_imgs, tst_labels) = dataset.load_data()
print (trn_imgs, trn_labels, tst_imgs, tst_labels)

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trn_imgs[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[trn_labels[i]])
plt.show()

mdl = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


mdl.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(epox)
mdl.fit(trn_imgs, trn_labels, epochs=epox, verbose=2)

loss, acc = mdl.evaluate(tst_imgs, tst_labels)
print("accuracy: ", acc)
print("loss: ", loss)

prds = mdl.predict(tst_imgs)

visualize(0)

