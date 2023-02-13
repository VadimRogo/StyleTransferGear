import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import PIL
import os
from PIL import Image
  

# create the root window
root = tk.Tk()
root.title('Tkinter Open File Dialog')
root.resizable(False, False)
root.geometry('300x150')
picturePath = ''
stylePath = ''


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_image(image_path, image_size=(2000, 1000)):
    img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def visualize(images, titles=('',)):
    noi = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w  * noi, w))
    grid_look = gridspec.GridSpec(1, noi, width_ratios=image_sizes)
    
    for i in range(noi):
        plt.subplot(grid_look[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i])
        plt.savefig("final.jpg")
    plt.show()

def visualizeFinal(images, titles=('',)):
    noi = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w  * noi, w))
    grid_look = gridspec.GridSpec(1, noi, width_ratios=image_sizes)
    
    for i in range(noi):
        plt.subplot(grid_look[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i])
        plt.savefig("final.jpg")
    plt.show()
    
def export_image(tf_img):
    tf_img = tf_img*255
    tf_img = np.array(tf_img, dtype=np.uint8)
    if np.ndim(tf_img)>3:
        assert tf_img.shape[0] == 1
        img = tf_img[0]
    return PIL.Image.fromarray(img)

def selectFile():
    global picturePath
    filetypes = (
        ('text files', '*.jpg'),
        ('All files', '*.*')
    )

    picturePath = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    print(picturePath)

def selectSecondFile():
    global stylePath
    filetypes = (
        ('text files', '*.jpg'),
        ('All files', '*.*')
    )

    stylePath = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    print(stylePath)

def mainDef():
    original_image = load_image(picturePath)
    style_image = load_image(stylePath)

    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')

    stylize_model = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    results = stylize_model(tf.constant(original_image), tf.constant(style_image))
    stylized_photo = results[0]

    # open method used to open different extension image file
    
    visualize([original_image, style_image, stylized_photo], titles=['Original Image', 'Style Image', 'Stylized Image'])
    visualizeFinal([stylized_photo], titles=['Stylized Image'])
# open button
openButton = ttk.Button(
    root,
    text='Choose picture',
    command=selectFile
)

openSecondButton = ttk.Button(
    root,
    text='Choose style',
    command=selectSecondFile
)

activateButton = tk.Button(text="Ready", command=mainDef)


openButton.pack(expand=True)
openSecondButton.pack(expand=True)
activateButton.pack(expand=True)

# run the application
root.mainloop()