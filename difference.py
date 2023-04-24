
from PIL import Image
from PIL.Image import Resampling
import numpy as np
import matplotlib.pyplot as plt
from math import *

images = []
def show_image(image_data, diff=False, title=None):
    images.append((image_data, diff, title))

# diffit = False
# if diffit:
#     show_diff(image1_data, image2_data)

# showboth = False
# if showboth:
#     plt.subplot(1, 2, 1)
#     plt.imshow(image1_data, cmap='gray')
#     plt.title('Q')
#     plt.subplot(1, 2, 2)
#     plt.imshow(image2_data, cmap='gray')
#     plt.title('O')
#     plt.show()

def show_diff(image1_data, image2_data, title=None):
    show_image(image1_data - image2_data, True, title)

def show_all_images():
    fig, axs = plt.subplots(1, len(images))
    for i in range(len(images)):
        image_data, diff, title = images[i]
        ax = axs[i] if len(images) > 1 else axs
        ax.set_title(title)
        if diff:
            ax.imshow(image_data, cmap='PiYG')
        else:
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
    padded_image_data = 255 - np.zeros(dimensions)
    padded_image_data[translate[1] : translate[1]+image_data.shape[0] , translate[0] : translate[0]+image_data.shape[1]] = image_data
    return padded_image_data


def image_rmse(image1_data, image2_data):
    mse = np.power(np.abs(image2_data - image1_data), 2).mean()
    rmse = sqrt(mse)
    return rmse

# translate is (x,y) as opposed to (row, column)
def display_translation_diff(image1_data, image2_data, translate, title=None):
    if translate[0] >= 0 and translate[1] >= 0:
        total_height = max(image1_data.shape[0], image2_data.shape[0] + translate[1])
        total_width = max(image1_data.shape[1], image2_data.shape[1] + translate[0])

        padded_image1_data = pad_translated(image1_data, (total_height, total_width))
        padded_image2_data = pad_translated(image2_data, (total_height, total_width), translate)

        show_diff(padded_image1_data, padded_image2_data)

    elif translate[0] >= 0 and translate[1] < 0:
        total_height = max(image1_data.shape[0] + -translate[1], image2_data.shape[0])
        total_width = max(image1_data.shape[1], image2_data.shape[1] + translate[0])

        padded_image1_data = pad_translated(image1_data, (total_height, total_width), (0, -translate[1]))
        padded_image2_data = pad_translated(image2_data, (total_height, total_width), (translate[0], 0))

        show_diff(padded_image1_data, padded_image2_data, title)

    else:
        display_translation_diff(image2_data, image1_data, (-translate[0], -translate[1]), title)

def display_scaled_translation_diff(image1_data, image2_data, im2_scale_factor, translate, title=None):
    scaled_image2 = scale(Image.fromarray(image2_data), im2_scale_factor)
    display_translation_diff(image1_data, np.array(scaled_image2), translate, title)


def rotate(xy, radians):
    r, theta = (hypot(xy[0], xy[1]), atan2(xy[1], xy[0]))
    theta += radians
    return (r * cos(theta), r * sin(theta))

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






# translate is (x,y) as opposed to (row, column)
# def translated_image_rmse2(image1_data, image2_data, translate, **kwargs):
#     print('inside translated_image_rmse2')
#     if translate[0] >= 0 and translate[1] >= 0:
#         print('inside translated_image_rmse2 - 1')
#         total_height = max(image1_data.shape[0], image2_data.shape[0] + translate[1])
#         total_width = max(image1_data.shape[1], image2_data.shape[1] + translate[0])

#         padded_image1_data = pad_translated(image1_data, (total_height, total_width))
#         padded_image2_data = pad_translated(image2_data, (total_height, total_width), translate)

#         print('showing diff2')
#         show_diff(padded_image1_data, padded_image2_data)
#         print('showing diff3')
#         return image_rmse(padded_image1_data, padded_image2_data)

#     if translate[0] >= 0 and translate[1] < 0:
#         print('inside translated_image_rmse2 - 2')
#         total_height = max(image1_data.shape[0] + -translate[1], image2_data.shape[0])
#         total_width = max(image1_data.shape[1], image2_data.shape[1] + translate[0])

#         padded_image1_data = pad_translated(image1_data, (total_height, total_width), (0, -translate[1]))
#         padded_image2_data = pad_translated(image2_data, (total_height, total_width), (translate[0], 0))

#         print('showing diff4')
#         show_diff(padded_image1_data, padded_image2_data)
#         print('showing diff5')
#         return image_rmse(padded_image1_data, padded_image2_data)

#     print('inside translated_image_rmse2 - 3')
#     return translated_image_rmse2(image2_data, image1_data, (-translate[0], -translate[1]))


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

# translate is (x,y) as opposed to (row, column) and translate is applied after scale
def scaled_translated_image_rmse3(image1_data, image2_data, im2_scale_factor, translate):
    image1_interpolator = RegularGridInterpolator((range(image1_data.shape[0]), range(image1_data.shape[1])), image1_data, bounds_error=False, fill_value=255)

    points = np.array([(i*im2_scale_factor[1] + translate[1], j*im2_scale_factor[0] + translate[0]) for i in range(image2_data.shape[0]) for j in range(image2_data.shape[1])])
    window_to_compare_to_image2 = image1_interpolator(points).reshape(image2_data.shape)

    # window_to_compare_to_image2 = np.vectorize(lambda i, j:
    #     image1_interpolator((i + translate[1], j + translate[0]))
    # )(range(image2_data.shape[0]), range(image2_data.shape[1]))

    return image_rmse(window_to_compare_to_image2, image2_data)

# translate is (x,y) as opposed to (row, column) and translate is applied after scale
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

    # window_to_compare_to_image2 = np.vectorize(lambda i, j:
    #     image1_interpolator((i + translate[1], j + translate[0]))
    # )(range(image2_data.shape[0]), range(image2_data.shape[1]))

    return image_rmse(window_to_compare_to_image2, image2_data)


# image1_filename = './ofl/images/FiraSans-Regular/g.png'
image1_filename = './ofl/images/Lexend[wght]/w.png'
# image2_filename = './ofl/images/OpenSans[wdth,wght]/g.png'
image2_filename = './ofl/handwrittenm.png'


image1 = Image.open(image1_filename)
image2 = Image.open(image2_filename)

# image1 = image1.resize((image1.size[0]//3, image1.size[1]//3))
# image2 = image2.resize((image2.size[0]//3, image2.size[1]//3))

# image1.thumbnail((75,75))
# image2.thumbnail((75,75))
image1.thumbnail((30,30))
image2.thumbnail((30,30))



image1_data = np.array(image1)
image2_data = np.array(image2)

image2_data = 255 - image2_data
image2_data = image2_data[np.any(image2_data, axis=1)]
image2_data = image2_data[:, np.any(image2_data, axis=0)]
image2_data = 255 - image2_data

show_image(image1_data)
show_image(image2_data)

# image1_data = pad(image1_data, (200, 200), (50, 10))
# image2_data = pad(image2_data, (200, 200), (20, 70))

# image1_data, image2_data = pad_to_same_size(image1_data, image2_data)
# show_diff(image1_data, image2_data)

# print(translated_image_rmse(image1_data, image2_data, (30, 40)))

import scipy.optimize as opt

def compare_translated_images(scale_rotate_translate):
    return scaled_rotated_translated_image_rmse3(image1_data, image2_data, scale_rotate_translate[0:2], scale_rotate_translate[2], scale_rotate_translate[3:])

heights_ratio = image1_data.shape[0] / image2_data.shape[0]
widths_ratio = image1_data.shape[1] / image2_data.shape[1]

# fit_ratio = min(heights_ratio, widths_ratio, 1)

bounds = [
    # (0, max(image2_data.shape[1], image1_data.shape[1])*2),
    # (0, max(image2_data.shape[0], image1_data.shape[0])*2)

    # (-image2_data.shape[1]//2, image1_data.shape[1]//2),
    # (-image2_data.shape[0]//2, image1_data.shape[0]//2)

    # (-image2_data.shape[1], image1_data.shape[1]),
    # (-image2_data.shape[0], image1_data.shape[0])

    # (0.8*min(heights_ratio, widths_ratio), 1.4*max(heights_ratio, widths_ratio)),
    (0.8*widths_ratio, 1.4*widths_ratio),
    (0.8*heights_ratio, 1.4*heights_ratio),
    (-pi/6, pi/6),
    (-image2_data.shape[1], image1_data.shape[1]),
    (-image2_data.shape[0], image1_data.shape[0])
]

# result = opt.differential_evolution(compare_translated_images, bounds, popsize=1000, init='sobol', integrality=(True, True), disp=True, workers=-1, x0=(0,0), strategy='randtobest1bin')

result = opt.differential_evolution(compare_translated_images, bounds, disp=True, workers=-1, popsize=50, init='sobol', polish=False, mutation=0.2)

# result = opt.brute(compare_translated_images, bounds, Ns=10, full_output=True)
# print(translated_image_rmse(image1_data, image2_data, (round(result.x[0]), round(result.x[1])), True))
# result = opt.dual_annealing(compare_translated_images, bounds, x0=(0,0))

# results = np.array([
#     opt.differential_evolution(compare_translated_images, bounds, disp=True, workers=-1, popsize=50) for i in range(22)
# ])
# xpoints = np.array([n for n in range(22)])
# ypoints = np.array([r.fun for r in results])
# plt.plot(xpoints, ypoints)
# plt.show()

# max_result = results[np.argmax(ypoints)]
# min_result = results[np.argmin(ypoints)]
# display_translation_diff(image1_data, image2_data, (floor(max_result.x[0]), floor(max_result.x[1])), f'max: {max_result.fun}')
# display_translation_diff(image1_data, image2_data, (floor(min_result.x[0]), floor(min_result.x[1])), f'min: {min_result.fun}')


print(result)

# display_translation_diff(image1_data, image2_data, (floor(result.x[0]), floor(result.x[1])))
# display_scaled_translation_diff(image1_data, image2_data, result.x[0], (floor(result.x[1]), floor(result.x[2])))

display_scaled_rotated_translation_diff(image1_data, image2_data, result.x[0:2], result.x[2], (floor(result.x[3]), floor(result.x[4])))

# display_scaled_rotated_translation_diff(image1_data, image2_data, 1, pi*8/6, (60,40))


show_all_images()

