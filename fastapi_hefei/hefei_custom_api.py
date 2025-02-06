import uvicorn
from fastapi import FastAPI
from loguru import logger
from fastapi_hefei.schemas import PostResultModel, ResponseModel, DetectModel, OCRModel
from tools.utils import check_file

from algorithm.yolov5_inference import Yolov5Infer
app = FastAPI()


@app.get("/")
async def root():
    return {"infer service for HeFei custom"}


@app.post(path="/algorithm/customs_pictogram_det_base64",
          response_model=ResponseModel,
          summary="象形图检测接口",
          description="象形图目标检测",
          tags=["目标检测"])
async def obj_detect_pictogram(pict_input: DetectModel):
    logger.info(pict_input)
    confidence = pict_input.det_threshold
    image_base64 = pict_input.image_base64
    image = check_file(image_base64)
    detector = Yolov5Infer(confidence=confidence)
    detector.set_infer_config(model_name="yolov5")
    reponse = detector.infer_yolov5(image)

    t = reponse[0].tolist()
    # todo: label to class
    results = []
    if len(reponse[0].tolist()) > 0:
        for box, score, label in zip(reponse[0].tolist(), reponse[1].tolist(), reponse[2].tolist()):
            result = PostResultModel(
                name=str(label),
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
    logger.info(warning_input)
    response = ResponseModel(success=True, message="success", result=[], code=200)
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
