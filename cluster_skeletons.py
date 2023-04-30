

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
import random
from sklearn.cluster import KMeans


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
            # ax.imshow(image_data, cmap='gray', vmin=0, vmax=255)
            ax.imshow(image_data, cmap='gray')
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


characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# characters = 'abc'

filenames = []
for dirname in os.listdir('ofl/images'):
    # print(dirname)
    for filename in os.listdir(f'ofl/images/{dirname}'):
        for character in characters:
            if filename == f'{character}.png':
                filenames.append(f'ofl/images/{dirname}/{filename}')
                break

np.random.shuffle(filenames)
# print('filenames:', filenames)


size = 32

def get_image_data(filename):
    image = Image.open(filename)
    # image.thumbnail((size, size), Image.Resampling.NEAREST)
    image.thumbnail((size-2, size-2), Image.Resampling.NEAREST)

    image_data = np.array(image)

    label = filename.split('/')[-1][0]
    return label, image_data

def convert_image(image_data):
    bw_image_data = bw_ify(image_data)

    bin_image_data = np.vectorize(lambda x: x < 255)(bw_image_data)
    bin_image_data = bin_image_data.copy(order='C')

    # skel_image_data = skeletonize(bin_image_data)
    skel_image_data = bin_image_data

    # crop
    skel_image_data = skel_image_data[np.any(skel_image_data, axis=1)]
    skel_image_data = skel_image_data[:, np.any(skel_image_data, axis=0)]

    # pad and center
    skel_image_data = pad_translated(skel_image_data, (size, size), ((size-skel_image_data.shape[1])//2, (size-skel_image_data.shape[0])//2))

    # skel_image_data = np.vectorize(lambda x: 1 if x else 0)(skel_image_data)

    return skel_image_data



processed_images = []
labels = []
for filename in filenames:
    label, data = get_image_data(filename)
    labels.append(label)
    converted_image = convert_image(data)
    processed_images.append(converted_image)


# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
# labels = enc.fit_transform(np.array(labels).reshape(-1, 1)).toarray()

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
labels = enc.fit_transform(labels)


# DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=5, activation="relu", padding="same", kernel_initializer="he_normal")

# model = tf.keras.Sequential([
#     DefaultConv2D(filters=64, kernel_size=7, input_shape=[size, size, 1]),
#     DefaultConv2D(filters=64, kernel_size=3, input_shape=[size, size, 1]),
#     tf.keras.layers.MaxPooling2D(pool_size=2),
#     DefaultConv2D(filters=128),
#     DefaultConv2D(filters=128),
#     tf.keras.layers.MaxPooling2D(pool_size=2),
#     DefaultConv2D(filters=256),
#     DefaultConv2D(filters=256),
#     tf.keras.layers.MaxPooling2D(pool_size=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
#     tf.keras.layers.Dense(units=8, activation="relu", kernel_initializer="he_normal"),
#     tf.keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
#     tf.keras.layers.Dense(units=len(characters), activation="softmax")
# ])
# print(model.summary())
# # tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# X_train = processed_images[:int(len(processed_images)*0.8)]
# y_train = labels[:int(len(processed_images)*0.8)]

# X_valid = processed_images[int(len(processed_images)*0.8):]
# y_valid = labels[int(len(processed_images)*0.8):]

# print(len(X_train), len(y_train), len(X_valid), len(y_valid))
# print(X_train[0])
# print(y_train[0])

# history = model.fit(np.array(X_train), np.array(y_train), epochs=18, validation_data=(np.array(X_valid), np.array(y_valid)))

# pd.DataFrame(history.history).plot(figsize=(8, 5), xlim=[0,29], ylim=[0,1], grid=True, xlabel='Epoch', ylabel='Accuracy', style=["r--", "r--.", "b-", "b-*"])
# plt.show()

# model.save('print_chars_model', save_format='tf')

# model = tf.keras.models.load_model('print_chars_model')


handwrittenm = convert_image(get_image_data('ofl/handwrittenm.png')[1])
# X_test = [handwrittenm[1]]
# y_test = [handwrittenm[0]]



# y_proba = model.predict(np.array(X_test))
# y_pred = y_proba.argmax(axis=-1)
# print(y_proba.round(2))
# print(y_pred)
# print(np.array([c for c in characters])[y_pred])


encoder = tf.keras.Sequential([
    tf.keras.layers.Dense(32*4),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(2)
])
decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(32*4),
    tf.keras.layers.Dense(32*32)
])
autoencoder = tf.keras.Sequential([encoder, decoder])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

# print(np.array(processed_images).shape)
# min_len = min([len(points) for points in processed_images])
# print(min_len)

# # shuffle each row of processed_images
# for i in range(len(processed_images)):
#     np.random.shuffle(processed_images[i])
# # drop rows to make all rows the same length
# normalized_images = np.array([points[:min_len].ravel() for points in processed_images])


# X_train = processed_images[:int(len(processed_images)*0.8)]
X_train = np.array(processed_images[:int(len(processed_images)*0.8)]).reshape(-1, 32*32)
X_valid = np.array(processed_images[int(len(processed_images)*0.8):]).reshape(-1, 32*32)
## X_train = np.array(X_train).reshape(-1, 32*32)
# X_train = normalized_images[:int(len(normalized_images)*0.8)]
y_train = labels[:int(len(processed_images)*0.8)]
y_valid = labels[int(len(processed_images)*0.8):]

history = autoencoder.fit(X_train, X_train, epochs=15, verbose=True, validation_data=(X_valid, X_valid))
encodings = encoder.predict(X_train)
print(encodings)
# codings = encoder.predict(X_train)

# pd.DataFrame(history.history).plot(figsize=(8, 5), xlim=[0,29], ylim=[0,1], grid=True, xlabel='Epoch', ylabel='Accuracy', style=["r--", "r--.", "b-", "b-*"])
# plt.show()

decodings = decoder.predict(encodings)

print(encodings.shape)
# print(codings[0])

# def points2shape(points):
#     buf = np.full((32, 32), False)
#     for point in points:
#         buf[point[0], point[1]] = True
#     return buf

# show_image(decodings[0].reshape(32, 32))
# show_image(X_train[0].reshape(32, 32))
show_image(X_train[1].reshape(32, 32))
show_image(decodings[1].reshape(32, 32))
show_image(handwrittenm.reshape(32, 32))

codings2 = encoder.predict(np.array([handwrittenm.ravel()]))
decodings2 = decoder.predict(codings2)
show_image(decodings2[0].reshape(32, 32))



# encoded_items = encoder_model(p_items)

# choose number of clusters K:
# sum_of_squared_distances = []
# K = range(1,30)
# for k in K:
#     km = KMeans(init='k-means++', n_clusters=k, n_init=10)
#     km.fit(codings)
#     sum_of_squared_distances.append(km.inertia_)

# plt.plot(K, sum_of_squared_distances, 'bx-')
# plt.vlines(ymin=0, ymax=150000, x=8, colors='red')
# plt.text(x=8.2, y=130000, s="optimal K=8")
# plt.xlabel('Number of Clusters K')
# plt.ylabel('Sum of squared distances')
# plt.title('Elbow Method For Optimal K')
# plt.show()

kmeans = KMeans(init='k-means++', n_clusters=62, n_init=10)
kmeans.fit(encodings)
P = kmeans.predict(encodings)

print(P[0])



# show_all_images()

# plot the clusters
plt.scatter(encodings[:, 0], encodings[:, 1], c=y_train, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

