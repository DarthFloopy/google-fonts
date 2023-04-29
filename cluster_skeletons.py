

from PIL import Image
from PIL.Image import Resampling
import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.interpolate import RegularGridInterpolator
from skimage.morphology import skeletonize
import tensorflow as tf
from functools import partial
import os
import pandas as pd


images = []
def show_image(image_data, diff=False, title=None):
    images.append((image_data, diff, title))

def show_diff(image1_data, image2_data, title=None):
    show_image(image1_data - image2_data, True, title)

def show_all_images():
    fig, axs = plt.subplots(1, len(images))
    for i in range(len(images)):
        image_data, diff, title = images[i]
        if image_data.dtype == bool:
            image_data = image_data.astype(np.uint8) * 255

        ax = axs[i] if len(images) > 1 else axs
        ax.set_title(title)
        if diff:
            ax.imshow(image_data, cmap='PiYG')
        else:
            ax.imshow(image_data, cmap='gray', vmin=0, vmax=255)
    plt.show()



def scale(image, factor, resample=Resampling.BICUBIC):
    return image.resize((int(image.width * factor[0]), int(image.height * factor[1])), resample=resample)


def pad_to_same_size(image1_data, image2_data):
    max_height = max(image1_data.shape[0], image2_data.shape[0])
    max_width = max(image1_data.shape[1], image2_data.shape[1])
    padded_image1_data = 255 - np.zeros((max_height, max_width))
    padded_image2_data = 255 - np.zeros((max_height, max_width))
    padded_image1_data[:image1_data.shape[0], :image1_data.shape[1]] = image1_data
    padded_image2_data[:image2_data.shape[0], :image2_data.shape[1]] = image2_data
    return padded_image1_data, padded_image2_data


def pad_translated(image_data, dimensions, translate=(0, 0)):
    # print(image_data.shape, dimensions, translate)

    # padded_image_data = 255 - np.zeros(dimensions)
    padded_image_data = np.full(dimensions, False)
    padded_image_data[translate[1] : translate[1]+image_data.shape[0] , translate[0] : translate[0]+image_data.shape[1]] = image_data
    return padded_image_data


# def distance(image1_data, image2_data):
#     points1 = np.argwhere(image1_data)
#     points2 = np.argwhere(image2_data)

#     distances = []

#     for p1 in points1:
#         closest = inf
#         for p2 in points2:
#             distance = hypot(p1[0]-p2[0], p1[1]-p2[1])
#             if distance < closest:
#                 closest = distance
#         distances.append(max(0,closest**2-1)/1)  # graph it in desmos to see

#     distances = np.power(distances, 2)

#     return sum(distances) / len(distances)

def distance(image1_data, image2_data):
    points1 = np.argwhere(image1_data)
    points2 = np.argwhere(image2_data)

    distances = []

    # for p1 in points1:
    #     closest = inf
    #     for p2 in points2:
    #         distance = hypot(p1[0]-p2[0], p1[1]-p2[1])
    #         if distance < closest:
    #             closest = distance
    #     distances.append(max(0,closest**2-1)/1)  # graph it in desmos to see

    # distances = np.power(distances, 2)

    for p1 in points1:
        # search in concentric squares for white pixels
        closest = inf
        for r in range(1, max(image2_data.shape[0], image2_data.shape[1])):
            if closest < inf:
                break

            for i in range(p1[0]-r, p1[0]+r+1):
                for j in [p1[1]-r, p1[1]+r]:
                    if j < 0 or j >= image2_data.shape[1]:  # could be optimized
                        continue
                    if i < 0 or i >= image2_data.shape[0]:
                        break

                    if image2_data[i,j]:
                        closest = min(closest, hypot(i-p1[0], j-p1[1]))

            for i in [p1[0]-r, p1[0]+r]:
                for j in range(p1[1]-r, p1[1]+r+1):
                    if j < 0 or j >= image2_data.shape[1]:
                        continue
                    if i < 0 or i >= image2_data.shape[0]:
                        break

                    if image2_data[i,j]:
                        closest = min(closest, hypot(i-p1[0], j-p1[1]))

        distances.append(closest ** 8)

    return sum(distances) / len(distances)



def bw_ify(image_data):
    return np.vectorize(lambda x: 0 if x < 255 else 255)(image_data)


filenames = []
for dirname in os.listdir('ofl/images'):
    print(dirname)
    for filename in os.listdir(f'ofl/images/{dirname}'):
        if filename.endswith('a.png') or filename.endswith('b.png') or filename.endswith('c.png'):
            filenames.append(f'ofl/images/{dirname}/{filename}')

print('filenames:', filenames)


size = 32

def convert_image(filename):
    image = Image.open(filename)
    # image.thumbnail((size, size), Image.Resampling.NEAREST)
    image.thumbnail((size-2, size-2), Image.Resampling.NEAREST)

    image_data = np.array(image)
    bw_image_data = bw_ify(image_data)

    bin_image_data = np.vectorize(lambda x: x < 255)(bw_image_data)
    bin_image_data = bin_image_data.copy(order='C')

    skel_image_data = skeletonize(bin_image_data)

    # crop
    skel_image_data = skel_image_data[np.any(skel_image_data, axis=1)]
    skel_image_data = skel_image_data[:, np.any(skel_image_data, axis=0)]

    # pad and center
    skel_image_data = pad_translated(skel_image_data, (size, size), ((size-skel_image_data.shape[1])//2, (size-skel_image_data.shape[0])//2))

    # skel_image_data = np.vectorize(lambda x: 1 if x else 0)(skel_image_data)

    label = filename.split('/')[-1][0]

    return label, skel_image_data

processed_images = []
labels = []
for dirname in filenames:
    processed_images.append(convert_image(dirname)[1])
    labels.append(convert_image(dirname)[0])

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
labels = enc.fit_transform(np.array(labels).reshape(-1, 1)).toarray()


DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal")

model = tf.keras.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[size, size, 1]),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=3, activation="softmax")
])
print(model.summary())
# tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

X_train = processed_images[:int(len(processed_images)*0.8)]
y_train = labels[:int(len(processed_images)*0.8)]

X_valid = processed_images[int(len(processed_images)*0.8):]
y_valid = labels[int(len(processed_images)*0.8):]

print(len(X_train), len(y_train), len(X_valid), len(y_valid))
print(X_train[0])
print(y_train[0])

history = model.fit(np.array(X_train), np.array(y_train), epochs=15, validation_data=(np.array(X_valid), np.array(y_valid)))


pd.DataFrame(history.history).plot(figsize=(8, 5), xlim=[0,29], ylim=[0,1], grid=True, xlabel='Epoch', ylabel='Accuracy', style=["r--", "r--.", "b-", "b-*"])
plt.show()


# show_image(skel_image1_data)

# show_all_images()

