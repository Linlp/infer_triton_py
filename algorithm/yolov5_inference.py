import tritonclient.http as httpclient
import numpy as np
import cv2


class Yolov5Infer(object):
    # url = "192.168.0.34:4010"
    # url = "192.168.0.33:8000"
    url = "0.0.0.0:8000"
    # model_name = "yolov5"
    client = httpclient.InferenceServerClient(url=url)

    __model_name = 'yolov5'
    __input_names = 'images'
    __output_names = 'output0'

    def __init__(self, confidence=0.25, iou=0.45):
        self.x_resize = 1.0
        self.y_resize = 1.0
        self.iou = iou
        self.confidence = confidence
        self.batch = False

    def set_batch(self, batch = True):
        self.batch = batch

    def set_url(self, ip="0.0.0.0", port=8000):
        global url, client
        url = f"{ip}:{port}"
        client = httpclient.InferenceServerClient(url=url)

    def set_infer_config(self, model_name='yolov5', input_names='images', output_names='output0'):
        self.__model_name = model_name
        self.__input_names = input_names
        self.__output_names = output_names

    def preprocess(self, image):
        image_resized = cv2.resize(image, (640, 640))
        image_rgb = image_resized[:, :, ::-1]  # BGR to RGB
        image_preprocessing = image_rgb.astype(np.float32) / 255.0  # normalize to [0, 1]
        image_preprocessing = np.transpose(image_preprocessing, (2, 0, 1))  # HWC to CHW

        # to do: add batch, channel, width, height: (1, 3, 640, 640)
        if self.batch:
            image_preprocessing = np.expand_dims(image_preprocessing, axis=0)

        x, y = image.shape[1], image.shape[0]
        self.x_resize, self.y_resize = float(x / 640), float(y / 640)
        return image_preprocessing

    def triton_infer(self, image_preprocessed, model_name="yolov5", input_name='images', output_name='output0'):
        yolov5_input = httpclient.InferInput(input_name, image_preprocessed.shape, "FP32")
        yolov5_input.set_data_from_numpy(image_preprocessed)
        response = self.client.infer(model_name=model_name, inputs=[yolov5_input])
        pred_results = response.as_numpy(output_name)
        return pred_results

    def postprocess(self, predictions, conf_threshold, iou_threshold):
        results = self.process_boxes(predictions, conf_threshold, iou_threshold)
        if len(results) > 0:
            boxes = results[:, :4].astype(int)
            confidences = results[:, 4]
            class_ids = results[:, 5].astype(int)
        else:
            boxes, confidences, class_ids = np.array([]), np.array([]), np.array([])
        return boxes, confidences, class_ids

    def process_boxes(self, predictions, conf_threshold, iou_threshold):
        # boxes = []
        # class_confidences = []
        # class_ids = []
        #
        # for detection in predictions[0]:
        #     conf = detection[4]
        #     scores = detection[5:]
        #     class_id = np.argmax(scores)
        #     class_confidence = scores[class_id]
        #     if conf > conf_threshold:
        #         box = detection[:4]
        #         boxes.append(box)
        #         class_confidences.append(class_confidence)
        #         class_ids.append(class_id)

        # 过滤置信度小于conf_threshold
        # (1, 25200, n) > (25200, n) , n:5+class
        detection = np.squeeze(predictions)
        conf = detection[..., 4]
        conf_mask = detection[..., 4] > conf_threshold
        detection = detection[conf_mask]
        boxes = detection[..., :4]
        scores = detection[..., 5:]
        class_ids = np.argmax(scores, axis=1)
        row_class_ids = np.arange(scores.shape[0])
        class_confidences = scores[row_class_ids, class_ids]
        # 置信度 = 前景置信度 * 目标分类置信度
        class_confidences = class_confidences * detection[..., 4]

        # list to numpy & reshape
        boxes = np.asarray(boxes, dtype=np.float32)
        class_confidences = np.asarray(class_confidences, dtype=np.float32).reshape(-1, 1)
        class_ids = np.asarray(class_ids, dtype=np.float32).reshape(-1, 1)

        if boxes.size == 0 or class_confidences.size == 0 or class_ids.size == 0:
            return np.array([])

        boxes = self.xywh2xyxy(boxes, self.x_resize, self.y_resize)
        detections = np.hstack((boxes, class_confidences, class_ids))
        # detections = np.concatenate((boxes, class_confidences, class_ids), axis=1)
        indexes = self.nms(detections, iou_threshold)
        detections = detections[indexes]
        return detections

    def nms(self, predictions, threshold):
        x1 = predictions[:, 0]
        y1 = predictions[:, 1]
        x2 = predictions[:, 2]
        y2 = predictions[:, 3]
        ares = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = predictions[:, 4]
        indexes = np.argsort(scores)[::-1]

        keep = []
        while indexes.size > 0:
            index = indexes[0]
            keep.append(index)
            x11 = np.maximum(x1[index], x1[indexes[1:]])
            y11 = np.maximum(y1[index], y1[indexes[1:]])
            x22 = np.minimum(x2[index], x2[indexes[1:]])
            y22 = np.minimum(y2[index], y2[indexes[1:]])
            w = np.maximum(0, x22 - x11)
            h = np.maximum(0, y22 - y11)
            overlaps = w * h
            ious = overlaps / (ares[index] + ares[indexes[1:]] - overlaps)
            idx = np.where(ious <= threshold)[0]
            indexes = indexes[idx + 1]
        return keep

    def xywh2xyxy(self, x, x_size, y_size):
        """
        x_center, y_center, width, height > xmin, ymin, xmax ,ymax
        """
        y = np.zeros_like(x)
        y[:, 0] = (x[:, 0] - x[:, 2] / 2) * x_size
        y[:, 1] = (x[:, 1] - x[:, 3] / 2) * y_size
        y[:, 2] = (x[:, 0] + x[:, 2] / 2) * x_size
        y[:, 3] = (x[:, 1] + x[:, 3] / 2) * y_size
        return y

    def infer_yolov5(self, image):
        image = self.preprocess(image)
        predictions = self.triton_infer(image, model_name=self.__model_name, input_name=self.__input_names,
                                        output_name=self.__output_names)
        #test
        # np.save("./triton_infer.npy", predictions)
        boxes, confs, ids = self.postprocess(predictions, self.confidence, self.iou)
        return boxes, confs, ids

    def drawbox(self):
        pass
