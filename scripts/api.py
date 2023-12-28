# api
from fastapi import FastAPI, Body
from pydantic import BaseModel
import numpy as np

from modules.api.models import *
from modules.api import api
from modules import paths_internal
from scripts.ocr_base import get_ocr_result
import gradio as gr


import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")



def ocr_api(_: gr.Blocks, app: FastAPI):
    class OcrResultItem(BaseModel):
        text: str
        box: list

    class OcrPredictResponse(BaseModel):
        result: list[OcrResultItem]
    
    class OcrPredictRequest(BaseModel):
        image: str

    @app.post("/ocr/predict", response_model=OcrPredictResponse)
    async def predict(request: OcrPredictRequest = Body(...)):
        dt_boxes,res,_ = get_ocr_result(request.image, is_api=True)
        result = [
            OcrResultItem(
                text=res[i][0],
                box=np.array(dt_boxes[i]).astype(np.int32).tolist()
            ) for i in range(len(dt_boxes))
        ]
        return OcrPredictResponse(result=result)
    
try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(ocr_api)
except:
    pass