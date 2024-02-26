from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel


model_id = "openai/clip-vit-base-patch32"
weights_dir = Path("./model_weights/")
weights_dir.mkdir(parents=True, exist_ok=True)


model = CLIPModel.from_pretrained(
    pretrained_model_name_or_path=model_id, cache_dir=weights_dir
)

processor = CLIPProcessor.from_pretrained(
    pretrained_model_name_or_path=model_id, cache_dir=weights_dir
)


# model.get_image_features()
# model.get_text_features()


def get_embedding():
    image = Image.open("./images/alc_logo.png")

    return processor(images=image, return_tensors="pt")
