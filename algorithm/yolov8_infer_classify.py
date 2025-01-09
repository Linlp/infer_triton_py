# from ultralytics import YOLO
# model = YOLO('./models/best.pt')
# path = './images/dogs/FnUB5DtlS9SkxVjR.jpg'
# results = model(path)
# probs = results[0].probs
#
# # pt转换onnx格式
# model.export(format="onnx")
# print(results)


import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import tritonclient.http as httpclient


class classify_infer_yolov8():
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def pre_process(self, image, onnx_flag=True):
        image = image / 255.0
        image = cv2.resize(image, (self.width, self.height))
        image = np.transpose(image,(2, 0, 1)) # to CHW
        if onnx_flag:
            image_array = np.expand_dims(image, axis=0)
        else:
            image_array = image
        return image_array

    def onnx_infer(self, image_array):
        session = ort.InferenceSession('./models/best.onnx')
        input_name = session.get_inputs()[0].name
        image_array = np.astype(image_array, np.float32)
        outputs = session.run(None, {input_name: image_array})
        output_data = outputs[0]
        return output_data

    def triton_infer(self, image_array):
        def triton_infer(image_preprocessed, model_name="yolov5", input_name='images', output_name='output0'):
            yolov5_input = httpclient.InferInput(input_name, image_preprocessed.shape, "FP32")
            yolov5_input.set_data_from_numpy(image_preprocessed)

            url = "192.168.0.33:8000"
            client = httpclient.InferenceServerClient(url=url)
            response = client.infer(model_name=model_name, inputs=[yolov5_input])
            pred_results = response.as_numpy(output_name)
            return pred_results

        image_array = image_array.astype( np.float32)
        output_data = triton_infer(image_array, model_name="classify_yolov8")
        return output_data


    def post_process(self, output, top_k):
        probs = np.squeeze(output) # (1, N) -> (N)
        top_indices = np.argsort(probs)[-top_k:][::-1]
        results = []
        for index in top_indices:
            result = (int(index), float(probs[index]))
            results.append(result)

        return results

    def classify_image(self, image, top_k, onnx_flag=True):
        image = self.pre_process(image, onnx_flag)
        if onnx_flag:
            output = self.onnx_infer(image)
        else:
            output = self.triton_infer(image)
        results = self.post_process(output, top_k)
        return results


def onnx_infer_classfy():
    image = Image.open('../datas/FnUB5DtlS9SkxVjR.jpg').convert('RGB')
    image = np.array(image)
    classify = classify_infer_yolov8(224, 224)
    results = classify.classify_image(image, 2)
    print(results)


def triton_infer_classfy():
    image = Image.open('../datas/FnUB5DtlS9SkxVjR.jpg').convert('RGB')
    image = np.array(image)
    classify = classify_infer_yolov8(224, 224)
    results = classify.classify_image(image, 2, False)
    print(results)


if __name__ == '__main__':
    triton_infer_classfy()
    #onnx_infer_classfy()

