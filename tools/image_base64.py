import sys

import cv2
import base64
import imghdr
import numpy as np


def get_image_format(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()

    image_format = imghdr.what(None, h=image_data)
    image_format = f".{image_format}"
    return image_format


def encode_image(image_path):
    image_format = get_image_format(image_path)
    image_array = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    _, buffer = cv2.imencode(image_format, image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image


def decode_image(image_base64):
    image_data = base64.b64decode(image_base64)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image
