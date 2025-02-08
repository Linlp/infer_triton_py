import base64
import cv2
import os
from tools.image_base64 import decode_image

def read_label_txt(path):
    labels = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            labels.append(line)
    return labels

def check_file(file):
    # image base64 or image path
    if os.path.exists(file):
        image = cv2.imread(file)
        return image
    elif isinstance(file, str) and len(file):
        try:
            base64.b64decode(file, validate=True)
            file = decode_image(file)
            return file
        except Exception as e:
            return None

    else:
        return None


def test_check_file():
    image = "../data/c2000_test.jpg"
    # image = encode_image(image)
    check = check_file(image)
    print(check)

def test_read_label_txt():
    image = "../labels/demo.txt"
    labels = read_label_txt(image)
    print(labels)

if __name__ == '__main__':
    test_read_label_txt()
    test_check_file()