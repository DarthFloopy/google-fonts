
from PIL import Image
from PIL.Image import Resampling
import numpy as np
import matplotlib.pyplot as plt
from math import *


fig = plt.figure()
first_time = True
def show_diff(image1_data, image2_data):
    print('showing diff')
    global first_time
    plt.imshow(image1_data - image2_data, cmap='PiYG')
    if first_time:
        plt.show()
        first_time = False
    else:
        fig.show()




def scale(image, factor, resample=Resampling.BICUBIC):
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
    # print(image_data.shape, dimensions, translate)
    padded_image_data = 255 - np.zeros(dimensions)
    padded_image_data[translate[1] : translate[1]+image_data.shape[0] , translate[0] : translate[0]+image_data.shape[1]] = image_data
    return padded_image_data


def image_rmse(image1_data, image2_data):
    mse = np.power(np.abs(image2_data - image1_data), 2).mean()
    rmse = sqrt(mse)
    return rmse

# translate is (x,y) as opposed to (row, column)
def translated_image_rmse2(image1_data, image2_data, translate, **kwargs):
    print('inside translated_image_rmse2')
    if translate[0] >= 0 and translate[1] >= 0:
        print('inside translated_image_rmse2 - 1')
        total_height = max(image1_data.shape[0], image2_data.shape[0] + translate[1])
        total_width = max(image1_data.shape[1], image2_data.shape[1] + translate[0])

        padded_image1_data = pad_translated(image1_data, (total_height, total_width))
        padded_image2_data = pad_translated(image2_data, (total_height, total_width), translate)

        print('showing diff2')
        show_diff(padded_image1_data, padded_image2_data)
        print('showing diff3')
        return image_rmse(padded_image1_data, padded_image2_data)

    if translate[0] >= 0 and translate[1] < 0:
        print('inside translated_image_rmse2 - 2')
        total_height = max(image1_data.shape[0] + -translate[1], image2_data.shape[0])
        total_width = max(image1_data.shape[1], image2_data.shape[1] + translate[0])

        padded_image1_data = pad_translated(image1_data, (total_height, total_width), (0, -translate[1]))
        padded_image2_data = pad_translated(image2_data, (total_height, total_width), (translate[0], 0))

        print('showing diff4')
        show_diff(padded_image1_data, padded_image2_data)
        print('showing diff5')
        return image_rmse(padded_image1_data, padded_image2_data)

    print('inside translated_image_rmse2 - 3')
    return translated_image_rmse2(image2_data, image1_data, (-translate[0], -translate[1]))

# translate is (x,y) as opposed to (row, column)
def translated_image_rmse(image1_data, image2_data, translate, **kwargs):
    if translate[0] >= 0 and translate[1] >= 0:
        total_height = max(image1_data.shape[0], image2_data.shape[0] + translate[1])
        total_width = max(image1_data.shape[1], image2_data.shape[1] + translate[0])

        padded_image1_data = pad_translated(image1_data, (total_height, total_width))
        padded_image2_data = pad_translated(image2_data, (total_height, total_width), translate)

        # print('showing diff - display =', kwargs.get('display', False))
        if kwargs.get('display', False):
            show_diff(padded_image1_data, padded_image2_data)
        return image_rmse(padded_image1_data, padded_image2_data)

    if translate[0] >= 0 and translate[1] < 0:
        total_height = max(image1_data.shape[0] + -translate[1], image2_data.shape[0])
        total_width = max(image1_data.shape[1], image2_data.shape[1] + translate[0])

        padded_image1_data = pad_translated(image1_data, (total_height, total_width), (0, -translate[1]))
        padded_image2_data = pad_translated(image2_data, (total_height, total_width), (translate[0], 0))

        # print('showing diff - display =', kwargs.get('display', False))
        if kwargs.get('display', False):
            show_diff(padded_image1_data, padded_image2_data)
        return image_rmse(padded_image1_data, padded_image2_data)

    return translated_image_rmse(image2_data, image1_data, (-translate[0], -translate[1]))

from scipy.interpolate import RegularGridInterpolator

# translate is (x,y) as opposed to (row, column)
def translated_image_rmse3(image1_data, image2_data, translate, **kwargs):
    image1_interpolator = RegularGridInterpolator((range(image1_data.shape[0]), range(image1_data.shape[1])), image1_data, bounds_error=False, fill_value=255)

    points = np.array([(i + translate[1], j + translate[0]) for i in range(image2_data.shape[0]) for j in range(image2_data.shape[1])])
    window_to_compare_to_image2 = image1_interpolator(points).reshape(image2_data.shape)

    # window_to_compare_to_image2 = np.vectorize(lambda i, j:
    #     image1_interpolator((i + translate[1], j + translate[0]))
    # )(range(image2_data.shape[0]), range(image2_data.shape[1]))

    return image_rmse(window_to_compare_to_image2, image2_data)



image1_filename = './ofl/images/OpenSans[wdth,wght]/D.png'
image2_filename = './ofl/images/FiraSans-Regular/t.png'

image1 = Image.open(image1_filename)
image2 = Image.open(image2_filename)

image1_data = np.array(image1)
image2_data = np.array(image2)

# image1_data = pad(image1_data, (200, 200), (50, 10))
# image2_data = pad(image2_data, (200, 200), (20, 70))

# image1_data, image2_data = pad_to_same_size(image1_data, image2_data)
# show_diff(image1_data, image2_data)

# print(translated_image_rmse(image1_data, image2_data, (30, 40)))

import scipy.optimize as opt

def compare_translated_images(translate):
    return translated_image_rmse3(image1_data, image2_data, translate)

bounds = [
    # (0, max(image2_data.shape[1], image1_data.shape[1])*2),
    # (0, max(image2_data.shape[0], image1_data.shape[0])*2)

    # (-image2_data.shape[1], image1_data.shape[1]),
    # (-image2_data.shape[0], image1_data.shape[0])

    (-image2_data.shape[1], image1_data.shape[1]),
    (-image2_data.shape[0], image1_data.shape[0])
]

def f(*x):
    pass

# result = opt.differential_evolution(compare_translated_images, bounds, popsize=1000, init='sobol', integrality=(True, True), disp=True, workers=-1, x0=(0,0), strategy='randtobest1bin')
result = opt.differential_evolution(compare_translated_images, bounds, disp=True, workers=-1, popsize=40, init='sobol')
# result = opt.brute(compare_translated_images, bounds, Ns=10, full_output=True)
# print(translated_image_rmse(image1_data, image2_data, (round(result.x[0]), round(result.x[1])), True))
# result = opt.dual_annealing(compare_translated_images, bounds, x0=(0,0))


# x0, fval, grid, Jout = result
# print(x0, fval)
print(result)
print('about to show')
# print(translated_image_rmse2(image1_data, image2_data, (floor(x0[0]), floor(x0[1])), display=True))
print(translated_image_rmse2(image1_data, image2_data, (floor(result.x[0]), floor(result.x[1]))))

# print(image_rmse(image1_data, image2_data))

diffit = False
if diffit:
    show_diff(image1_data, image2_data)

showboth = False
if showboth:
    plt.subplot(1, 2, 1)
    plt.imshow(image1_data, cmap='gray')
    plt.title('Q')
    plt.subplot(1, 2, 2)
    plt.imshow(image2_data, cmap='gray')
    plt.title('O')
    plt.show()



