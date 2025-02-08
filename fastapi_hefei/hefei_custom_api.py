import uvicorn
from fastapi import FastAPI
from loguru import logger
from fastapi_hefei.schemas import PostResultModel, ResponseModel, DetectModel, OCRModel
from tools.utils import check_file, read_label_txt

from algorithm.yolov5_inference import Yolov5Infer
app = FastAPI()

import pathlib, os
label_path = pathlib.Path(__file__).parent.parent.joinpath('labels')
pictogram_labels = read_label_txt(os.path.join(label_path, 'demo.txt'))


def common_infer(input: DetectModel, labels_dict:dict):
    logger.info(input.serial_number)
    confidence = input.det_threshold
    image_base64 = input.image_base64
    image = check_file(image_base64)
    detector = Yolov5Infer(confidence=confidence)
    detector.set_infer_config(model_name="yolov5")
    response = detector.infer_yolov5(image)

    t = response[0].tolist()
    results = []
    if len(response[0].tolist()) > 0:
        for box, score, label in zip(response[0].tolist(), response[1].tolist(), response[2].tolist()):
            result = PostResultModel(
                # name=str(label),
                name=labels_dict[int(label)],
                confidence=score,
                xmin=box[0],
                ymin=box[1],
                xmax=box[2],
                ymax=box[3],
                title=None,
                note=None
            )
            results.append(result)

    response = ResponseModel(success=True, message="success", result=results, code=200)
    return response

@app.get("/")
async def root():
    return {"infer service for HeFei custom"}


@app.post(path="/algorithm/customs_pictogram_det_base64",
          response_model=ResponseModel,
          summary="象形图检测接口",
          description="象形图目标检测",
          tags=["目标检测"])
async def obj_detect_pictogram(pict_input: DetectModel):
    logger.info(pict_input.serial_number)
    confidence = pict_input.det_threshold
    image_base64 = pict_input.image_base64
    image = check_file(image_base64)
    detector = Yolov5Infer(confidence=confidence)
    detector.set_infer_config(model_name="yolov5")
    response = detector.infer_yolov5(image)

    t = response[0].tolist()
    results = []
    if len(response[0].tolist()) > 0:
        for box, score, label in zip(response[0].tolist(), response[1].tolist(), response[2].tolist()):
            result = PostResultModel(
                # name=str(label),
                name=pictogram_labels[int(label)],
                confidence=score,
                xmin=box[0],
                ymin=box[1],
                xmax=box[2],
                ymax=box[3],
                title=None,
                note=None
            )
            results.append(result)

    response = ResponseModel(success=True, message="success", result=results, code=200)
    return response


@app.post(path="/algorithm/customs_warning_det_base64",
          response_model=ResponseModel,
          summary="警示图检测接口",
          description="警示图目标检测",
          tags=["目标检测"])
async def obj_detect_warning(warning_input: DetectModel):
    response = common_infer(warning_input, pictogram_labels)
    return response


@app.post("/algorithm/character_rec_base64",
          response_model=ResponseModel,
          summary="ocr文字识别接口",
          description="ppv4 ocr检测",
          tags=["ocr识别"])
async def ppv4_ocr_rec(ocr_input: OCRModel):
    logger.info(ocr_input)
    response = ResponseModel(success=True, message="success", result=[], code=200)
    return response


if __name__ == "__main__":
    from config.port import IP, PORT
    # uvicorn.run(app, host="0.0.0.0", port=6610)
    uvicorn.run(app, host=IP, port=PORT)
