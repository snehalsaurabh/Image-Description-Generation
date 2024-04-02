from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
import io
import json
import requests

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {"max_length":max_length,"num_beams":num_beams}


# image pre-processing
def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)
    # received encoders above

    # passing the pixels through the model for decoding
    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

app = FastAPI(title="Image Description",description="API - For Generation of Image Description")
class ImageCaption(BaseModel):
        caption: str

@app.get("/",include_in_schema=False)
def index():
    return HTMLResponse(url="/docs")

