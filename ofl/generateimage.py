
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# command used to generate font_filenames.txt
# (run under ofl directory under project root)
# find . -name '*.ttf'| grep -Ev -e '-[^R]' > font_filenames.txt

font_files = []
with open("./font_filenames.txt", "r") as f:
    font_files = f.read().splitlines()
print(font_files)
print(len(font_files))




FONT_SIZE = 100
MAX_PADDING = 0
def generate_image(text, font_path):
    font_object = ImageFont.truetype(font_path, FONT_SIZE) # Font has to be a .ttf file
    word = "W"

    fg = "#000000"  # black foreground
    bg = "#FFFFFF"  # white background

    text_width, text_height = font_object.getsize(word)
    image = Image.new('RGBA', (text_width + MAX_PADDING*2, text_height + MAX_PADDING*2), color=bg)
    draw_pad = ImageDraw.Draw(image)

    draw_pad.text((MAX_PADDING, MAX_PADDING-6), word, font=font_object, fill=fg)

    image = image.convert("L") # Use this if you want to binarize image
    return image

def crop(array):
    array = 255 - array
    array = array[np.any(array, axis=1)]
    array = array[:, np.any(array, axis=0)]
    array = 255 - array
    return array



image = generate_image('W', "./spacegrotesk/SpaceGrotesk[wght].ttf")

data = np.array(image)
# print(data[len(data)//2])

data = crop(data)

# convert the array back to an image
image = Image.fromarray(data)




# plot the image in matplotlib
plt.imshow(data, cmap='rainbow')
plt.show()

# data = data / 255.0

file_name = "output.png"

image.save(file_name)

