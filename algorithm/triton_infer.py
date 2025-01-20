import tritonclient.http as httpclient
import numpy as np
import platform


url = ""
os_name = platform.system()
if os_name == "Windows":
    url = "192.168.0.33:8000"
elif os_name == "Linux":
    url = "0.0.0.0:8000"

client = httpclient.InferenceServerClient(url=url)


def triton_infer(image_preprocessed, model_name="yolov5", input_name='images', output_name='output0'):
    yolov5_input = httpclient.InferInput(input_name, image_preprocessed.shape, "FP32")
    yolov5_input.set_data_from_numpy(image_preprocessed)
    response = client.infer(model_name=model_name, inputs=[yolov5_input])
    pred_results = response.as_numpy(output_name)
    return pred_results


def ocr_det_infer(image_preprocessed, model_name="pp_v4_det", input_name='x', output_name='sigmoid_11.tmp_0'):
    yolov5_input = httpclient.InferInput(input_name, image_preprocessed.shape, "FP32")
    yolov5_input.set_data_from_numpy(image_preprocessed)
    response = client.infer(model_name=model_name, inputs=[yolov5_input])
    pred_results = response.as_numpy(output_name)
    return pred_results


def ocr_rec_infer(image_preprocessed, model_name="pp_v4_rec", input_name='x', output_name='softmax_2.tmp_0'):
    yolov5_input = httpclient.InferInput(input_name, image_preprocessed.shape, "FP32")
    yolov5_input.set_data_from_numpy(image_preprocessed)
    response = client.infer(model_name=model_name, inputs=[yolov5_input])
    pred_results = response.as_numpy(output_name)
    return pred_results


def ocr_det_t1():
    input_data = np.random.rand(1, 3, 960, 960).astype(np.float32)
    output_data = ocr_det_infer(input_data)
    print(output_data)


def ocr_rec_t1():
    input_data = np.random.rand(1, 3, 48, 640).astype(np.float32)
    output_data = ocr_rec_infer(input_data)
    print(output_data)


if __name__ == '__main__':
    # ocr_det_t1()
    ocr_rec_t1()
    # input_data = np.random.rand(3, 224, 224).astype(np.float32)
    # output_data = triton_infer(input_data, model_name="classify_yolov8")
    # print(output_data)
