

from functools import partial
# from keras.engine.topology import Layer, InputSpec
from keras import backend as K
import keras
from keras import layers
from keras import callbacks
from keras.initializers import VarianceScaling
from keras.layers import Dense, Input
from keras.layers import Layer, InputSpec
from keras.models import Model
from keras.optimizers import SGD
from math import *
from PIL.Image import Resampling
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from skimage.morphology import skeletonize
from sklearn.cluster import KMeans
from time import time
import keras.backend as K
import matplotlib.pyplot as plt
import metrics
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as linear_assignment  # fix for old code

# copied from https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/
# class ClusteringLayer(Layer):
#     """
#     Clustering layer converts input sample (feature) to soft label.

#     # Example
#     ```
#         model.add(ClusteringLayer(n_clusters=10))
#     ```
#     # Arguments
#         n_clusters: number of clusters.
#         weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
#         alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
#     # Input shape
#         2D tensor with shape: `(n_samples, n_features)`.
#     # Output shape
#         2D tensor with shape: `(n_samples, n_clusters)`.
#     """

#     def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
#         if 'input_shape' not in kwargs and 'input_dim' in kwargs:
#             kwargs['input_shape'] = (kwargs.pop('input_dim'),)
#         super(ClusteringLayer, self).__init__(**kwargs)
#         self.n_clusters = n_clusters
#         self.alpha = alpha
#         self.initial_weights = weights
#         self.input_spec = InputSpec(ndim=2)

#     def build(self, input_shape):
#         assert len(input_shape) == 2
#         input_dim = input_shape[1]
#         self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
#         self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
#         if self.initial_weights is not None:
#             self.set_weights(self.initial_weights)
#             del self.initial_weights
#         self.built = True

#     def call(self, inputs, **kwargs):
#         """ student t-distribution, as same as used in t-SNE algorithm.        
#                  q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
#                  q_ij can be interpreted as the probability of assigning sample i to cluster j.
#                  (i.e., a soft assignment)
#         Arguments:
#             inputs: the variable containing data, shape=(n_samples, n_features)
#         Return:
#             q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
#         """
#         q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
#         q **= (self.alpha + 1.0) / 2.0
#         q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
#         return q

#     def compute_output_shape(self, input_shape):
#         assert input_shape and len(input_shape) == 2
#         return input_shape[0], self.n_clusters

#     def get_config(self):
#         config = {'n_clusters': self.n_clusters}
#         base_config = super(ClusteringLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

# def autoencoder(dims, act='relu', init='glorot_uniform'):
#     """
#     Fully connected auto-encoder model, symmetric.
#     Arguments:
#         dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
#             The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
#         act: activation, not applied to Input, Hidden and Output layers
#     return:
#         (ae_model, encoder_model), Model of autoencoder and model of encoder
#     """
#     n_stacks = len(dims) - 1
#     # input
#     input_img = Input(shape=(dims[0],), name='input')
#     x = input_img
#     # internal layers in encoder
#     for i in range(n_stacks-1):
#         x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

#     # hidden layer
#     encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

#     x = encoded
#     # internal layers in decoder
#     for i in range(n_stacks-1, 0, -1):
#         x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

#     # output
#     x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
#     decoded = x
#     return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')




# end copied code

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


def points_distance(image1_data, image2_data):
    points1 = np.argwhere(image1_data)
    points2 = np.argwhere(image2_data)

    p2_distances_by_p1 = {}

    for p1 in points1:
        p1_tuple = tuple(p1)
        distances = []
        for p2 in points2:
            p2_tuple = tuple(p2)
            distance = hypot(p1[0]-p2[0], p1[1]-p2[1])
            distances.append((p2, distance))
            distances.sort(key=lambda x: x[1])
        p2_distances_by_p1[tuple(p1)] = distances

    closest_match = sorted(p2_distances_by_p1.values(), key=lambda p2_distance_pairs: p2_distance_pairs[0][1])[0][0]  # (p2, distance)
    # print(p2_distances_by_p1)
    # print(closest_match)

    return closest_match

    initial_index = points1.index(closest_match[0])
    for i in range(len(points1)):
        index = (initial_index + i) % len(points1)

    # return sum(closest_matches) / len(closest_matches)

def weighted_avg_closest_pixel(image1_data, image2_data):
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
# characters = 'abcdefg'

# filenames = []
# for dirname in os.listdir('ofl/images'):
#     # print(dirname)
#     for filename in os.listdir(f'ofl/images/{dirname}'):
#         for character in characters:
#             if filename == f'{character}.png':
#                 filenames.append(f'ofl/images/{dirname}/{filename}')
#                 break

# np.random.shuffle(filenames)
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

    skel_image_data = skeletonize(bin_image_data)
    # skel_image_data = bin_image_data

    # crop
    skel_image_data = skel_image_data[np.any(skel_image_data, axis=1)]
    skel_image_data = skel_image_data[:, np.any(skel_image_data, axis=0)]

    # pad and center
    skel_image_data = pad_translated(skel_image_data, (size, size), ((size-skel_image_data.shape[1])//2, (size-skel_image_data.shape[0])//2))

    # skel_image_data = np.vectorize(lambda x: 1 if x else 0)(skel_image_data)

    # if np.any(np.isnan(skel_image_data)):
    #     print('nan', skew_image_data)
    #     raise Exception('nan')
    return skel_image_data


handwrittenm = get_image_data('ofl/handwrittenm.png')[1]
converted_handwrittenm = convert_image(get_image_data('ofl/handwrittenm.png')[1])
# show_image(handwrittenm)
# show_image(converted_handwrittenm)









def rotate(xy, radians):
    r, theta = (hypot(xy[0], xy[1]), atan2(xy[1], xy[0]))
    theta += radians
    return (r * cos(theta), r * sin(theta))

# currently in use
# Note: this doesn't give pixel-perfect accuracy (since it's interpolating the scaling backwards), but it works to visualize what is going on
def display_scaled_rotated_translation_diff(image1_data, image2_data, im2_scale_factor, im2_rotation, translate, title=None):
    # scaled_image2 = scale(Image.fromarray(image2_data), im2_scale_factor)
    # rotated_image2 = scaled_image2.rotate(degrees(im2_rotation), fillcolor=255)
    # # padded_image1 = pad_translated(image1_data, (max(image1_data.shape[0], image2_data.shape[0]*cos)), ())
    # display_translation_diff(image1_data, np.array(rotated_image2), translate, title)

    # perform the following steps to process the transforms:
    # 1. create a square buffer with dimensions 3 times as long as the largest dimension of either image
    # 2. create a RegularGridInterpolator for image2_data
    # 3. for each pixel in the buffer, calculate the corresponding pixel in image2_data by applying the inverse of the transforms to the pixel's coordinate
    # 4. use the RegularGridInterpolator to get that pixel value from image2_data and store it in the buffer.
    # 5. crop the buffer
    biggest_dimension = max(image1_data.shape[0], image2_data.shape[0], image1_data.shape[1], image2_data.shape[1])
    buf_size = int(biggest_dimension*3*max(im2_scale_factor[0], im2_scale_factor[1], 1))
    buf = 255 - np.zeros((buf_size, buf_size))

    x = np.arange(0, image2_data.shape[1])
    y = np.arange(0, image2_data.shape[0])
    f = RegularGridInterpolator((y, x), image2_data, bounds_error=False, fill_value=255, method='linear')
    for i in range(buf.shape[0]):
        for j in range(buf.shape[1]):
            # calculate the corresponding pixel in image2_data
            pixel = (i-buf_size//2, j-buf_size//2)

            # then, translate the pixel's coordinate by -translate
            pixel = (pixel[0] - translate[1], pixel[1] - translate[0])

            # first, rotate the pixel's coordinate by -im2_rotation
            pixel = rotate(pixel, -im2_rotation)

            # then, scale the pixel's coordinate by 1/im2_scale_factor
            pixel = (pixel[0]/im2_scale_factor[1], pixel[1]/im2_scale_factor[0])

            # finally, round the pixel's coordinate to the nearest integer
            pixel = (floor(pixel[0]), floor(pixel[1]))

            # add the offset back
            # pixel = (pixel[0] + int(biggest_dimension*1.5), pixel[1] + int(biggest_dimension*1.5))

            # print(pixel)
            # get the pixel value from image2_data
            try:
                buf[i, j] = f(pixel)
                # print('pixel in bounds: {}'.format(pixel))
            except:
                print('pixel out of bounds: {}'.format(pixel))
                print('image2_data.shape: {}'.format(image2_data.shape))
                return


    # draw image1_data
    im1_x = buf_size//2
    im1_y = buf_size//2
    print(f'im1_x: {im1_x}, im1_y: {im1_y}, image1_data.shape: {image1_data.shape}')

    buf = 255 - buf
    image1_data = 255 - image1_data
    buf[im1_x:im1_x+image1_data.shape[0], im1_y:im1_y+image1_data.shape[1]] -= image1_data
    # buf = 255 - buf

    # buf[buf_size//2:buf_size//2+image1_data.shape[0], buf_size//2:buf_size//2+image1_data.shape[1]] = image1_data

    # crop the buffer
    # buf = 255 - buf
    buf = buf[np.any(buf, axis=1)]
    buf = buf[:, np.any(buf, axis=0)]
    buf = 255 - buf

    # display the difference
    show_image(buf, True)


def scaled_rotated_translated_image_rmse3(image1_data, image2_data, im2_scale_factor, im2_rotation, translate):
    image1_interpolator = RegularGridInterpolator((range(image1_data.shape[0]), range(image1_data.shape[1])), image1_data, bounds_error=False, fill_value=255)

    # points = np.array([(i*im2_scale_factor + translate[1], j*im2_scale_factor + translate[0]) for i in range(image2_data.shape[0]) for j in range(image2_data.shape[1])])

    points = np.array([
        rotate((i*im2_scale_factor[1], j*im2_scale_factor[0]), im2_rotation)
            for i in range(image2_data.shape[0]) for j in range(image2_data.shape[1])
    ])
    # translate each point in points
    points = np.array([(i + translate[1], j + translate[0]) for i, j in points])

    window_to_compare_to_image2 = image1_interpolator(points).reshape(image2_data.shape)

    ## old code that never worked
    # window_to_compare_to_image2 = np.vectorize(lambda i, j:
    #     image1_interpolator((i + translate[1], j + translate[0]))
    # )(range(image2_data.shape[0]), range(image2_data.shape[1]))

    # return points_distance(window_to_compare_to_image2, image2_data)
    return points_distance(window_to_compare_to_image2, image2_data)



def set_images(image1_input, image2_input):
    global image1, image2
    image1 = image1_input
    image2 = image2_input


# image1 = image1.resize((image1.size[0]//3, image1.size[1]//3))
# image2 = image2.resize((image2.size[0]//3, image2.size[1]//3))

# image1.thumbnail((75,75))
# image2.thumbnail((75,75))
def init_images():
    global image1, image2, image1_data, image2_data

    image1_data = np.array(image1)
    image1_data = skeletonize(image1_data)
    image1 = Image.fromarray(image1_data)
    image1.thumbnail((30,30))
    image1_data = np.array(image1)


    image2_data = np.array(image2)

    # crop
    image2_data = 255 - image2_data
    image2_data = image2_data[np.any(image2_data, axis=1)]
    image2_data = image2_data[:, np.any(image2_data, axis=0)]
    image2_data = 255 - image2_data

    image2 = Image.fromarray(image2_data)
    image2.thumbnail((30,30))
    image2_data = np.array(image2)





import scipy.optimize as opt

def compare_translated_images(scale_rotate_translate):
    return scaled_rotated_translated_image_rmse3(image1_data, image2_data, scale_rotate_translate[0:2], scale_rotate_translate[2], scale_rotate_translate[3:])

def calculate_result():
    heights_ratio = image1_data.shape[0] / image2_data.shape[0]
    widths_ratio = image1_data.shape[1] / image2_data.shape[1]

    # fit_ratio = min(heights_ratio, widths_ratio, 1)

    # intervals to check for each feature with differential evolution optimization
    bounds = [
        # (0, max(image2_data.shape[1], image1_data.shape[1])*2),
        # (0, max(image2_data.shape[0], image1_data.shape[0])*2)

        # (-image2_data.shape[1]//2, image1_data.shape[1]//2),
        # (-image2_data.shape[0]//2, image1_data.shape[0]//2)

        # (-image2_data.shape[1], image1_data.shape[1]),
        # (-image2_data.shape[0], image1_data.shape[0])

        # (0.8*min(heights_ratio, widths_ratio), 1.4*max(heights_ratio, widths_ratio)),
        (0.8*widths_ratio, 1.4*widths_ratio),  # x scaling
        (0.8*heights_ratio, 1.4*heights_ratio),  # y scaling
        (-pi/4, pi/4),  # rotation
        # (0, pi*2),
        (-image2_data.shape[1], image1_data.shape[1]),  # x translation
        (-image2_data.shape[0], image1_data.shape[0])  # y translation
    ]

    # result = opt.differential_evolution(compare_translated_images, bounds, popsize=1000, init='sobol', integrality=(True, True), disp=True, workers=-1, x0=(0,0), strategy='randtobest1bin')

    # the hard work is done by scipy here
    result = opt.differential_evolution(compare_translated_images, bounds, disp=True, workers=-1, popsize=100, init='sobol', polish=False, mutation=0.45, recombination=0.99)
    return result

image1_filename = './ofl/images/Lexend[wght]/m.png'
# image2_filename = './ofl/images/IBMPlexSans-Regular/m.png'
image2_filename = './ofl/handwrittenm.png'

image1 = Image.open(image1_filename)
image2 = Image.open(image2_filename)

set_images(image1, image2)
init_images()

print(points_distance(image1_data, image2_data))
raise SystemExit()
result = calculate_result()

print(result)

show_image(image1_data)
show_image(image2_data)

display_scaled_rotated_translation_diff(image1_data, image2_data, result.x[0:2], result.x[2], (floor(result.x[3]), floor(result.x[4])))

# processed_images = []
# labels = []
# for filename in filenames:
#     label, data = get_image_data(filename)
#     labels.append(label)
#     converted_image = convert_image(data)
#     processed_images.append(converted_image)
    
# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
# labels = enc.fit_transform(np.array(labels).reshape(-1, 1)).toarray()

# from sklearn.preprocessing import LabelEncoder
# enc = LabelEncoder()
# labels = enc.fit_transform(labels)


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


# handwrittenm = convert_image(get_image_data('ofl/handwrittenm.png')[1])
# X_test = [handwrittenm[1]]
# y_test = [handwrittenm[0]]



# y_proba = model.predict(np.array(X_test))
# y_pred = y_proba.argmax(axis=-1)
# print(y_proba.round(2))
# print(y_pred)
# print(np.array([c for c in characters])[y_pred])


# encoder = tf.keras.Sequential([
#     # tf.keras.layers.Conv2D(filters=64, kernel_size=7, input_shape=[32,32,1]),
#     # tf.keras.layers.Conv2D(filters=64, kernel_size=3),
#     # tf.keras.layers.MaxPooling2D(pool_size=2),
#     # tf.keras.layers.Flatten(),
#     tf.keras.layers.Flatten(input_shape=[32,32]),
#     # tf.keras.layers.BatchNormalization(),
#     # tf.keras.layers.Dense(8, activation="relu", kernel_initializer="he_normal"),
#     # tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
#     # tf.keras.layers.Dense(16, activation="relu", kernel_initializer="he_normal"),

#     # tf.keras.layers.Dropout(0.5),
#     # tf.keras.layers.Dense(32*32, activation="relu", kernel_initializer="he_normal"),
#     # tf.keras.layers.Reshape((32, 32)),
#     # tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same", kernel_initializer="he_normal"),
#     # tf.keras.layers.Flatten(),


#     tf.keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
# ])
# decoder = tf.keras.Sequential([
#     # tf.keras.layers.Dense(16, activation="relu", kernel_initializer="he_normal"),
#     # tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
#     # tf.keras.layers.Dropout(0.5),
#     # tf.keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
#     # tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(32*32, activation="relu", kernel_initializer="he_normal"),
#     tf.keras.layers.Reshape((32, 32)),
# ])
# autoencoder = tf.keras.Sequential([encoder, decoder])

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.0009)
# # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# # autoencoder.compile(optimizer="RMSprop", loss='mse', metrics=['accuracy'])
# autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

# # This is the size of our encoded representations
# encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# # This is our input image
# input_img = keras.Input(shape=(32*32,))
# # "encoded" is the encoded representation of the input
# encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# # "decoded" is the lossy reconstruction of the input
# decoded = layers.Dense(32*32, activation='sigmoid')(encoded)

# # This model maps an input to its reconstruction
# autoencoder = keras.Model(input_img, decoded)
# # Let's also create a separate encoder model:

# # This model maps an input to its encoded representation
# encoder = keras.Model(input_img, encoded)
# # As well as the decoder model:

# # This is our encoded (32-dimensional) input
# encoded_input = keras.Input(shape=(encoding_dim,))
# # Retrieve the last layer of the autoencoder model
# decoder_layer = autoencoder.layers[-1]
# # Create the decoder model
# decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
# # Now let's train our autoencoder to reconstruct MNIST digits.

# # First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adam optimizer:

# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# print(np.array(processed_images).shape)
# min_len = min([len(points) for points in processed_images])
# print(min_len)

# # shuffle each row of processed_images
# for i in range(len(processed_images)):
#     np.random.shuffle(processed_images[i])
# # drop rows to make all rows the same length
# normalized_images = np.array([points[:min_len].ravel() for points in processed_images])

# num_images = len(processed_images)
# split = int(num_images*0.8)

# X_train = processed_images[:int(len(processed_images)*0.8)]

# X_train = np.array(processed_images[:split]).reshape(split, 32*32)
# X_train = np.array(processed_images)#.reshape(split, 32*32)

#if anything in X_train is NaN, print it
# for i in range(len(X_train)):
#     if np.any(np.isnan(X_train[i])):
#         print(f'X_train[{i}]',X_train[i])

# X_valid = np.array(processed_images[split:]).reshape(num_images-split, 32*32)

#if anything in X_valid is NaN, print it
# for i in range(len(X_valid)):
#     if np.any(np.isnan(X_valid[i])):
#         print(f'X_valid[{i}]',X_valid[i])


## X_train = np.array(X_train).reshape(-1, 32*32)
# X_train = normalized_images[:int(len(normalized_images)*0.8)]



# y_train = labels[:split]
# y_valid = labels[split:]
# y_train = labels
# y_valid = labels[split:]

# history = autoencoder.fit(X_train, X_train, epochs=100, verbose=True, shuffle=True, validation_data=(X_valid, X_valid))
# encodings = encoder.predict(X_train)

# history_fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# history_fig.suptitle('Autoencoder Training Performance')
# ax1.plot(range(0,100), history.history['accuracy'], color='blue')
# ax1.set(ylabel='Reconstruction Accuracy')
# ax2.plot(range(0,100), np.log10(history.history['loss']), color='blue')
# ax2.plot(range(0,100), np.log10(history.history['val_loss']), color='red', alpha=0.9)
# ax2.set(ylabel='log_10(loss)', xlabel='Training Epoch')
# plt.show()
# raise SystemExit()
# print('encodings', encodings)
# codings = encoder.predict(X_train)

# pd.DataFrame(history.history).plot(figsize=(8, 5), xlim=[0,29], ylim=[0,1], grid=True, xlabel='Epoch', ylabel='Accuracy', style=["r--", "r--.", "b-", "b-*"])
# plt.show()

# decodings = decoder.predict(encodings)

# print(encodings.shape)
# print(codings[0])

# def points2shape(points):
#     buf = np.full((32, 32), False)
#     for point in points:
#         buf[point[0], point[1]] = True
#     return buf

# show_image(decodings[0].reshape(32, 32))
# show_image(X_train[0].reshape(32, 32))
# show_image(X_train[1].reshape(32, 32))
# show_image(decodings[1].reshape(32, 32))
# show_image(handwrittenm.reshape(32, 32))

# codings2 = encoder.predict(np.array([handwrittenm.ravel()]))
# decodings2 = decoder.predict(codings2)
# show_image(decodings2[0].reshape(32, 32))



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


# kmeans = KMeans(init='k-means++', n_clusters=62, n_init=10)
# kmeans.fit(encodings)
# P = kmeans.predict(encodings)

# print(P[0])

# n_clusters = len(characters)
# x = np.concatenate((X_train, X_valid))
# y = np.concatenate((y_train, y_valid))
# # x = X_train
# # y = y_train
# index_array = np.arange(x.shape[0])
# index = 0
# # copied code
# # dims = [x.shape[-1], 500, 500, 2000, 10]
# # init = VarianceScaling(scale=1. / 3., mode='fan_in',
#                            # distribution='uniform')
# # pretrain_optimizer = SGD(lr=1, momentum=0.9)
# # pretrain_epochs = 300
# batch_size = 256
# # save_dir = './results'









# from time import time
# import numpy as np
# import keras.backend as K
# from keras.layers import Layer, InputSpec
# from keras.layers import Dense, Input
# from keras.models import Model
# from keras.optimizers import SGD
# from keras import callbacks
# from keras.initializers import VarianceScaling
# from sklearn.cluster import KMeans
# import metrics

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2, random_state=42)

# x = np.concatenate((x_train, x_test))
# y = np.concatenate((y_train, y_test))
# x = x.reshape((x.shape[0], -1))
# # x = np.divide(x, 255.)
# n_clusters = len(np.unique(y))
# print(x.shape)

# pretrain_optimizer = SGD(lr=1, momentum=0.9)
# pretrain_epochs = 50
# batch_size = 256
# save_dir = './mlresults'

# dims = [x.shape[-1], 500, 500, 2000, 64]
# init = VarianceScaling(scale=1. / 3., mode='fan_in',
#                            distribution='uniform')

# autoencoder, encoder = autoencoder(dims, init=init)

# from keras.utils import plot_model
# # plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)

# autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
# # autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)
# # autoencoder.save_weights(save_dir + '/ae_weights.h5')
# autoencoder.load_weights(save_dir + '/ae_weights.h5')

# clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
# model = Model(inputs=encoder.input, outputs=clustering_layer)

# model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

# # Initialize cluster centers using k-means.
# kmeans = KMeans(n_clusters=n_clusters, n_init=20)
# y_pred = kmeans.fit_predict(encoder.predict(x))
# y_pred_last = np.copy(y_pred)
# model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])


# def target_distribution(q):
#     weight = q ** 2 / q.sum(0)
#     return (weight.T / weight.sum(1)).T


# loss = 0
# index = 0
# maxiter = 8000
# update_interval = 140
# index_array = np.arange(x.shape[0])
# # tol = 0.001
# tol = 0.003

# for ite in range(int(maxiter)):
#     if ite % update_interval == 0:
#         q = model.predict(x, verbose=0)
#         p = target_distribution(q)  # update the auxiliary target distribution p
#         # evaluate the clustering performance
#         y_pred = q.argmax(1)
#         if y is not None:
#             acc = np.round(metrics.acc(y, y_pred), 5)
#             nmi = np.round(metrics.nmi(y, y_pred), 5)
#             ari = np.round(metrics.ari(y, y_pred), 5)
#             loss = np.round(loss, 5)
#             print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

#         # check stop criterion - model convergence
#         delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
#         y_pred_last = np.copy(y_pred)
#         if ite > 0 and delta_label < tol:
#             print('delta_label ', delta_label, '< tol ', tol)
#             print('Reached tolerance threshold. Stopping training.')
#             break

#     idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
#     loss = model.train_on_batch(x=x[idx], y=p[idx])
#     index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

# model.save_weights(save_dir + '/DEC_model_final.h5')

# model.load_weights(save_dir + '/DEC_model_final.h5')

# Eval.
# q = model.predict(x, verbose=0)
# p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
# y_pred = q.argmax(1)
# if y is not None:
#     acc = np.round(metrics.acc(y, y_pred), 5)
#     nmi = np.round(metrics.nmi(y, y_pred), 5)
#     ari = np.round(metrics.ari(y, y_pred), 5)
#     loss = np.round(loss, 5)
#     print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)


# import seaborn as sns
# import sklearn.metrics
# import matplotlib.pyplot as plt
# sns.set(font_scale=3)
# confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)

# plt.figure(figsize=(16, 14))
# sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
# plt.title("Confusion matrix", fontsize=30)
# plt.ylabel('True label', fontsize=25)
# plt.xlabel('Clustering label', fontsize=25)
# plt.show()

# encoded_images = model.predict(x_train)
# print('predicting')
# decoded_images = autoencoder.predict(x[:5])
# handwrittenm = handwrittenm.reshape(1, 1, 1024)
# decoded_handwrittenm = autoencoder.predict(handwrittenm)

# for i in range(5):
#     show_image(x[i].reshape(32,32))
#     show_image(decoded_images[i].reshape(32,32))
# show_image(handwrittenm.reshape(32,32))
# show_image(decoded_handwrittenm.reshape(32,32))
# show_image(X_train[0].reshape(32, 32))
# show_image(X_train[1].reshape(32, 32))
# show_image(decodings[1].reshape(32, 32))
# show_image(handwrittenm.reshape(32, 32))

# codings2 = encoder.predict(np.array([handwrittenm.ravel()]))
# decodings2 = decoder.predict(codings2)
# show_image(decodings2[0].reshape(32, 32))





# #
# y_true = y.astype(np.int64)
# D = max(y_pred.max(), y_true.max()) + 1
# w = np.zeros((D, D), dtype=np.int64)
# # Confusion matrix.
# for i in range(y_pred.size):
#     w[y_pred[i], y_true[i]] += 1
# ind = linear_assignment(-w)
# ind = np.asarray(ind)
# ind = np.transpose(ind)

# acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
# print('acc', acc)


# end copied code


show_all_images()

# # plot the clusters
# plt.scatter(encodings[:, 0], encodings[:, 1], c=y_train, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=1)
# plt.show()

# visualize the clusters:
# from mpl_toolkits.mplot3d import Axes3D
# encoded_fig = plt.figure()
# ax = Axes3D(encoded_fig)
# # p = ax.scatter(encodings[:,0], encodings[:,1], encodings[:,2], c=y_pred, marker="o", picker=True, cmap="rainbow")
# p = ax.scatter(encodings[:,0], encodings[:,1], encodings[:,2])
# plt.colorbar(p, shrink=0.5)
# plt.show()

