# ocr base
import os
from infer.utility import get_my_dir
from infer.predict import TextPredict
import numpy as np
import cv2
from modules.api import api
from modules import paths_internal

models_dir = os.path.join(paths_internal.models_path, "ocr")

def get_args():
    return {
        'use_gpu': True, 
        'image_dir': None, 
        'det_algorithm': 'DB', 
        'det_model_dir': models_dir, 
        'det_limit_side_len': 960, 
        'det_limit_type': 'max', 
        'det_box_type': 'quad', 
        'det_db_thresh': 0.3, 
        'det_db_box_thresh': 0.6, 
        'det_db_unclip_ratio': 1.5, 
        'max_batch_size': 10, 
        'use_dilation': False, 
        'det_db_score_mode': 'fast', 
        'rec_algorithm': 'SVTR_LCNet', 
        'rec_model_dir': models_dir,
        'rec_image_inverse': True, 
        'rec_image_shape': '3, 48, 320', 
        'rec_batch_num': 6, 
        'max_text_length': 25, 
        'vis_font_path': os.path.join(get_my_dir(),"doc/fonts/simfang.ttf"), 
        'drop_score': 0.5, 
        'use_angle_cls': False, 
        'cls_model_dir': None, 
        'cls_image_shape': '3, 48, 192', 
        'label_list': ['0', '180'], 
        'cls_batch_num': 6, 
        'cls_thresh': 0.9, 
        'warmup': False, 
        'output': './inference_results', 
        'save_crop_res': False, 
        'crop_res_save_dir': './output', 
        'use_mp': False, 
        'total_process_num': 1, 
        'process_id': 0, 
        'show_log': True
    }

def get_ocr_result(img,is_api=False):
    args = get_args()
    # convert args to argparse.Namespace
    args = type('args', (object,), args)
    ocr = TextPredict(args)
    if is_api:
        imgdata = api.decode_base64_to_image(img)
    else:
        imgdata = img
    # convert imgdata to cv2 image
    imgdata = cv2.cvtColor(np.asarray(imgdata), cv2.COLOR_RGB2BGR)
    return ocr(imgdata)

