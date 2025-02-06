from pydantic import BaseModel,Field
from typing import Optional, List


class PostResultModel(BaseModel):
    name: str = Field(examples=["BY-B1550"], title="识别文字", description="识别出的文字/目标")
    confidence: float = Field(examples=[0.99646246], title="置信度", description="置信度")
    xmin: int = Field(examples=[40], title="目标识别框左上角 x 坐标", description="目标识别框左上角 x 坐标")
    ymin: int = Field(examples=[18], title="目标识别框左上角 y 坐标", description="目标识别框左上角 y 坐标")
    xmax: int = Field(examples=[129], title="目标识别框右下角 x 坐标", description="目标识别框右下角 x 坐标")
    ymax: int = Field(examples=[39], title="目标识别框右下角 y 坐标", description="目标识别框右下角 y 坐标")
    title: Optional[str] = Field(examples=["商品名称"], title="查验名称", description="查验名称")
    note: Optional[str] = Field(examples=[None], title="备注说明", description="备注说明")


class ResponseModel(BaseModel):
    success: bool = Field(example=True, title="成功标识", description="成功标识")
    message: str = Field(examples=["图像 base64-通用文字识别, 检测完成"], title="文字说明", description="文字说明")
    result: List[PostResultModel] = Field(title="目标识别框和识别的文字", description="目标识别框和识别的文字")
    code: int = Field(examples=[200], title="消息代码", description="消息代码，200 为成功，其他为失败")


class DetectModel(BaseModel):
    serial_number: str = Field(examples=["BY-B1550"], title="识别文字", description="识别出的文字/目标")
    det_threshold: float = Field(examples=[0.99646246], title="置信度", description="置信度")
    image_base64: str = Field(examples=[40], title="待识别图像数据", description="base64格式图片数据")


class OCRModel(BaseModel):
    serial_number: int = Field(examples=[1], title="识别文字", description="识别出的文字/目标")
    character_type: int = Field(examples=[3], title="识别文字", description="识别出的文字/目标")
    char_det_thres: float = Field(examples=[0.33], title="检测置信度", description="字符检测置信度")
    char_rec_thres: float = Field(examples=[0.33], title="识别置信度", description="字符识别置信度")
    image_base64: str = Field(examples=[40], title="待识别图像数据", description="base64格式图片数据")