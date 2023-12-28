# ui
from modules import script_callbacks, paths_internal
from scripts.deoldify_base import *
import gradio as gr
import tempfile
import os
import shutil
from scripts.ocr_base import get_ocr_result

def process_image(img):
    _, rec_res, _ = get_ocr_result(img)
    return "\n".join([str(r) for r in rec_res])

def ocr_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tab("OCR"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="识别图片",type="pil")
                with gr.Column():
                    output = gr.Textbox(lines=25, readonly=True, elem_id="log_area")
            run_button = gr.Button(label="Run")
            run_button.click(inputs=[input_image],outputs=[output],fn=process_image)
    return [(ui,"OCR","OCR")]

script_callbacks.on_ui_tabs(ocr_tab)