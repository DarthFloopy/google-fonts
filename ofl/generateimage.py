
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os

characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# command used to generate font_filenames.txt
# (run under ofl directory under project root)
# find . -name '*.ttf'| grep -Ev -e '-[^R]' > font_filenames.txt

font_files = []
with open("./font_filenames.txt", "r") as f:
    font_files = f.read().splitlines()




FONT_SIZE = 100
MAX_PADDING = 40
def generate_image(text, font_path):
    font_object = ImageFont.truetype(font_path, FONT_SIZE) # Font has to be a .ttf file

    fg = "#000000"  # black foreground
    bg = "#FFFFFF"  # white background

    text_width, text_height = font_object.getsize(text)
    image = Image.new('RGBA', (text_width + MAX_PADDING*2, text_height + MAX_PADDING*2), color=bg)
    draw_pad = ImageDraw.Draw(image)

    draw_pad.text((MAX_PADDING, MAX_PADDING-6), text, font=font_object, fill=fg)
    # draw again but with spacing between letters
    draw_pad.text((MAX_PADDING, MAX_PADDING-6), text, font=font_object, fill=fg, spacing=10)


    image = image.convert("L") # Use this if you want to binarize image
    return image

def crop(array):
    array = 255 - array
    array = array[np.any(array, axis=1)]
    array = array[:, np.any(array, axis=0)]
    array = 255 - array
    return array


# create folder for each font
for font_file in font_files:
    font_name = font_file.split("/")[-1].split(".")[0]
    folder_name = f"images/{font_name}"
    os.makedirs(folder_name, exist_ok=True)

for font_file in font_files:
    font_name = font_file.split("/")[-1].split(".")[0]
    for char in characters:
        image = generate_image(char, font_file)
        data = np.array(image)
        data = crop(data)
        image = Image.fromarray(data)

        file_name = f"images/{font_name}/{char}.png"
        try:
            image.save(file_name)
        except:
            print(f"Error saving {file_name}")
            break


# plot the image in matplotlib
# plt.imshow(data, cmap='rainbow')
# plt.show()

