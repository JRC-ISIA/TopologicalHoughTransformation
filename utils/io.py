import pickle

from PIL import Image


def load_wireframe_data(file_path):
    # Function to load the .pkl file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_image_as_grayscale(image_path, width, height):
    # Open the image from the specified path
    image = Image.open(image_path)

    # Convert the image to grayscale ('L' mode)
    grayscale_image = image.convert('L')

    # Resize the image to the specified dimensions
    resized_image = grayscale_image.resize((width, height))

    return resized_image
