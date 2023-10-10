

# To run this, you'll need to get some images to compare, such as the generated images of fonts' characters, and mess with filename paths so the script will find them.
# Link to discord message where I posted a generated images tar: https://discord.com/channels/@me/1092879603300827228/1099429390326169671
# You'll also need to install a few packages (see imports)
# See the bottom of the script for basic usage of the functions

from math import *
from PIL.Image import Resampling
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

# matplotlib doesn't like being called multiple times in a program so we are just queueing everything for later
images = []

def show_image(image_data, diff=False, title=None):
    images.append((image_data, diff, title))

def show_diff(image1_data, image2_data, title=None):
    show_image(image1_data - image2_data, True, title)

# show_all_images() is called at the end of the script to display everything in the queue
def show_all_images():
    fig, axs = plt.subplots(1, len(images))
    for i in range(len(images)):
        image_data, diff, title = images[i]
        ax = axs[i] if len(images) > 1 else axs
        ax.set_title(title)
        if diff:
            ax.imshow(image_data, cmap='PiYG', vmin=0, vmax=512)
        else:
            ax.imshow(image_data, cmap='gray')
    plt.show()



def scale(image, factor, resample=Resampling.BICUBIC):
    return image.resize((int(image.width * factor[0]), int(image.height * factor[1])), resample=resample)

def pad_translated(image_data, dimensions, translate=(0, 0)):
    # print(image_data.shape, dimensions, translate)
    padded_image_data = 255 - np.zeros(dimensions)
    padded_image_data[translate[1] : translate[1]+image_data.shape[0] , translate[0] : translate[0]+image_data.shape[1]] = image_data
    return padded_image_data


def image_rmse(image1_data, image2_data):
    mse = np.power(image2_data - image1_data, 2).mean()
    rmse = sqrt(mse)
    return rmse

def rotate(xy, radians):
    r, theta = (hypot(xy[0], xy[1]), atan2(xy[1], xy[0]))
    theta += radians
    return (r * cos(theta), r * sin(theta))

# currently in use
# Note: this doesn't give pixel-perfect accuracy (since it's interpolating the scaling backwards), but it works to visualize what is going on
def display_scaled_rotated_translation_diff(image1_data, image2_data, im2_scale_factor, im2_rotation, translate, title=None):
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


    # crop the buffer
    # buf = 255 - buf
    buf = buf[np.any(buf, axis=1)]
    buf = buf[:, np.any(buf, axis=0)]
    buf = 255 - buf

    # display the difference
    show_image(buf, True)






# translate is (x,y) as opposed to (row, column) and translate is applied after scale
def scaled_rotated_translated_image_rmse3(image1_data, image2_data, im2_scale_factor, im2_rotation, translate):
    image1_interpolator = RegularGridInterpolator((range(image1_data.shape[0]), range(image1_data.shape[1])), image1_data, bounds_error=False, fill_value=255)

    points = np.array([
        rotate((i*im2_scale_factor[1], j*im2_scale_factor[0]), im2_rotation)
            for i in range(image2_data.shape[0]) for j in range(image2_data.shape[1])
    ])
    # translate each point in points
    points = np.array([(i + translate[1], j + translate[0]) for i, j in points])

    window_to_compare_to_image2 = image1_interpolator(points).reshape(image2_data.shape)

    return image_rmse(window_to_compare_to_image2, image2_data)


def set_images(image1_input, image2_input):
    global image1, image2
    image1 = image1_input
    image2 = image2_input


def init_images():
    global image1, image2, image1_data, image2_data

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



def compare_translated_images(scale_rotate_translate):
    return scaled_rotated_translated_image_rmse3(image1_data, image2_data, scale_rotate_translate[0:2], scale_rotate_translate[2], scale_rotate_translate[3:])

def calculate_result():
    heights_ratio = image1_data.shape[0] / image2_data.shape[0]
    widths_ratio = image1_data.shape[1] / image2_data.shape[1]

    # intervals to check for each feature with differential evolution optimization
    bounds = [
        (0.6*widths_ratio, 1.5*widths_ratio),  # x scaling
        (0.6*heights_ratio, 1.5*heights_ratio),  # y scaling
        (-pi/7, pi/7),  # rotation
        (-image2_data.shape[1], image1_data.shape[1]),  # x translation
        (-image2_data.shape[0], image1_data.shape[0])  # y translation
    ]

    # result = opt.differential_evolution(compare_translated_images, bounds, popsize=1000, init='sobol', integrality=(True, True), disp=True, workers=-1, x0=(0,0), strategy='randtobest1bin')

    # the hard work is done by scipy here
    result = opt.differential_evolution(compare_translated_images, bounds, disp=True, workers=-1, popsize=200, init='sobol', polish=False, mutation=0.15, recombination=0.95)
    return result

image1_filename = './ofl/images/Lexend[wght]/m.png'
image2_filename = './ofl/images/FiraSans-Regular/n.png'
# image2_filename = './ofl/handwrittenm.png'

image1 = Image.open(image1_filename)
image2 = Image.open(image2_filename)

set_images(image1, image2)
init_images()
result = calculate_result()

print(result)

show_image(image1_data)
show_image(image2_data)

display_scaled_rotated_translation_diff(image1_data, image2_data, result.x[0:2], result.x[2], (floor(result.x[3]), floor(result.x[4])))

show_all_images()

