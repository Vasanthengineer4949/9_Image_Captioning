import config
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel 
from fastapi import FastAPI, UploadFile
import uvicorn

device = ["cuda" if torch.cuda.is_available() else "cpu"][0]

app = FastAPI(debug=True)
encoder = config.ENCODER_CKPT
decoder = config.DECODER_CKPT
captioner_ckpt = config.HUB_PUSHED_ID
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder)
tokenizer = AutoTokenizer.from_pretrained(decoder)
captioner = VisionEncoderDecoderModel.from_pretrained(captioner_ckpt).to(device)

@app.get("/")
def home():
    return {"Project Name": "Image Captioning"}

@app.post("/uploadimage/")
async def upload_image(image_file: UploadFile):
    url = config.IMG_PATH + "/inference_images/" + image_file.filename
    image = Image.open(url)
    clean_text = lambda x: x.replace("<|endoftext|>", "").split("\n")[0]
    sample = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    caption_ids = captioner.generate(sample, max_length=100)[0]
    caption = clean_text(tokenizer.decode(caption_ids))
    return caption
    
if __name__ == "__main__":
    uvicorn.run(app)
