
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from math import *


def show_diff(image1_data, image2_data):
    plt.imshow(image1_data - image2_data, cmap='PiYG')
    plt.show()




def scale(image, factor, resample=Image.BICUBIC):
    return image.resize((int(image.width * factor), int(image.height * factor)), resample=resample)


def pad_to_same_size(image1_data, image2_data):
    max_height = max(image1_data.shape[0], image2_data.shape[0])
    max_width = max(image1_data.shape[1], image2_data.shape[1])
    padded_image1_data = 255 - np.zeros((max_height, max_width))
    padded_image2_data = 255 - np.zeros((max_height, max_width))
    padded_image1_data[:image1_data.shape[0], :image1_data.shape[1]] = image1_data
    padded_image2_data[:image2_data.shape[0], :image2_data.shape[1]] = image2_data
    return padded_image1_data, padded_image2_data


def pad_translated(image_data, dimensions, translate=(0, 0)):
    print(image_data.shape, dimensions, translate)
    padded_image_data = 255 - np.zeros(dimensions)
    padded_image_data[translate[1] : translate[1]+image_data.shape[0] , translate[0] : translate[0]+image_data.shape[1]] = image_data
    return padded_image_data


def image_rmse(image1_data, image2_data):
    mse = np.power(np.abs(image1_data - image2_data), 2).mean()
    rmse = np.sqrt(mse)
    return rmse


# translate is (x,y) as opposed to (row, column)
def translated_image_rmse(image1_data, image2_data, translate):
    if translate[0] >= 0 and translate[1] >= 0:
        total_height = max(image1_data.shape[0], image2_data.shape[0] + translate[1])
        total_width = max(image1_data.shape[1], image2_data.shape[1] + translate[0])

        padded_image1_data = pad_translated(image1_data, (total_height, total_width))
        padded_image2_data = pad_translated(image2_data, (total_height, total_width), translate)
        # show_diff(padded_image1_data, padded_image2_data)
        return image_rmse(padded_image1_data, padded_image2_data)

    if translate[0] >= 0 and translate[1] < 0:
        total_height = max(image1_data.shape[0] + -translate[1], image2_data.shape[0])
        total_width = max(image1_data.shape[1], image2_data.shape[1] + translate[0])

        padded_image1_data = pad_translated(image1_data, (total_height, total_width), (0, -translate[1]))
        padded_image2_data = pad_translated(image2_data, (total_height, total_width), (translate[0], 0))
        # show_diff(padded_image1_data, padded_image2_data)
        return image_rmse(padded_image1_data, padded_image2_data)

    return translated_image_rmse(image2_data, image1_data, (-translate[0], -translate[1]))






image1_filename = './ofl/images/FiraSans-Regular/Q.png'
image2_filename = './ofl/images/FiraSans-Regular/O.png'

image1 = Image.open(image1_filename)
image2 = Image.open(image2_filename)




image1_data = np.array(image1)
image2_data = np.array(image2)




# image1_data = pad(image1_data, (200, 200), (50, 10))
# image2_data = pad(image2_data, (200, 200), (20, 70))

image1_data, image2_data = pad_to_same_size(image1_data, image2_data)


# print(image1_data[0])
# print(image2_data[0])
# print()

# print(image1_data.shape)
# print(image2_data.shape)

# square each vaule in image1_data and image2_data



# print(translated_image_rmse(image1_data, image2_data, (30, 40)))



# image1_data = np.array(scale(image1, 0.5))
# image2_data = np.array(scale(image1, 0.75, resample=Image.LANCZOS))
# image1_data, image2_data = pad_to_same_size(image1_data, image2_data)

diffit = True

if diffit:
    show_diff(image1_data, image2_data)
else:
    plt.subplot(1, 2, 1)
    plt.imshow(image1_data, cmap='gray')
    plt.title('Q')
    plt.subplot(1, 2, 2)
    plt.imshow(image2_data, cmap='gray')
    plt.title('O')
    plt.show()



