'''
Image Augmentation

how to pre-process the images before training our model,
by using 'ImageDataGenerator' method to convert <a simple image> to become <an uncommon image>
for an our strong model performance
'''

# Plot of images as baseline for comparison
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os

from keras import backend as K

K.set_image_dim_ordering('th')

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# -------------------------------------------------
# define data preparation
'''
ZCA: https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening
'''
datagen = ImageDataGenerator()  # default
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)  # ปรับสีให้ชัด?
datagen = ImageDataGenerator(zca_whitening=True)  # เน้นขอบ
datagen = ImageDataGenerator(rotation_range=90)  # หมุนภาพ
datagen = ImageDataGenerator(width_shift_range=.2, height_shift_range=.2)  # ย้ายตำแหน่งภาพ
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)  # ภาพสลับด้าน
# -------------------------------------------------

# fit parameters from data
datagen.fit(X_train)

# configure batch size and retrieve one batch of images
# * alternative for save img (can plot and save in same time by using save_to_dir, etc.)
os.makedirs('./chapter_20/images')
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir='./chapter_20/images',
                                     save_prefix='aug', save_format='png'):
    # create a grid of 3x3 images
    for i in range(0, 9):
        # now the shape of X_batch is (9, 1, 28, 28)
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()
    break