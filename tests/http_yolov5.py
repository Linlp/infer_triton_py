import requests
from tools.image_base64 import encode_image
from config.port import SERVER_URL

def t_http_yolov5():
    url = f'{SERVER_URL}/algorithm/customs_pictogram_det_base64'
    image = encode_image('../datas/c2000_yolov5.png')
    data = {
        "serial_number": "123456",
        "det_threshold": 0.5,
        "image_base64": image
    }

    response = requests.post(url, json=data)
    print(response.text)


if __name__ == '__main__':
    t_http_yolov5()
