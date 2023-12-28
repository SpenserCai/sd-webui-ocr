import os
import launch
from modules import paths_internal

from huggingface_hub import snapshot_download

models_dir = os.path.join(paths_internal.models_path, "ocr")

# Download the model
model_name = "spensercai/ppocr_v4_torch"

# check model exists
def check_model_exists(model_name):
    if os.path.exists(os.path.join(models_dir, model_name)):
        return True
    else:
        return False
    
if not check_model_exists("ch_PP-OCRv4_server_det.onnx") or not check_model_exists("ch_PP-OCRv4_server_rec.onnx"):
    snapshot_download(repo_id="spensercai/ppocr_v4_torch",local_dir_use_symlinks=False, local_dir=models_dir,allow_patterns="*.onnx")

for dep in ['onnxruntime-gpu', 'tensorboard', 'opencv-python', 'attrdict', 'pyyaml', 'openpyxl', 'premailer', 'rapidfuzz', 'pyclipper', 'imgaug', 'shapely']:
     if not launch.is_installed(dep):
        launch.run_pip(f"install {dep}", f"{dep} for sd-webui-ocr extension")


