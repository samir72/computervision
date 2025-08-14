from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import Iterable
import os
import sys

CONF_THRESHOLD = 0.50  # 50% confidence

@dataclass
class Config:
    endpoint: str
    key: str
    project_id: str
    published_name: str

def load_config() -> Config:
    # .env support if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    endpoint = os.getenv("ObjectPredictionEndpoint", "").rstrip("/")
    key = os.getenv("ObjectPredictioncredential", "")
    project_id = os.getenv("Objectprojectid", "")
    published_name = os.getenv("ObjModelName", "")  # This must be the *published name*

    missing = [k for k, v in {
        "ObjectPredictionEndpoint": endpoint,
        "ObjectPredictioncredential": key,
        "Objectprojectid": project_id,
        "ObjModelName (published name)": published_name
    }.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return Config(endpoint, key, project_id, published_name)

def get_client(cfg: Config) -> CustomVisionPredictionClient:
    creds = ApiKeyCredentials(in_headers={"Prediction-key": cfg.key})
    return CustomVisionPredictionClient(endpoint=cfg.endpoint, credentials=creds)

def detect_image(client: CustomVisionPredictionClient, cfg: Config, image_path: str):
    with open(image_path, "rb") as f:
        return client.detect_image(cfg.project_id, cfg.published_name, f)

def annotate_image(image_path: str, predictions: Iterable, out_path: str) -> int:
    """
    Draw bounding boxes and labels for predictions over threshold.
    Returns the number of drawn detections.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    # Try to load a readable font; fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, w // 50))
    except Exception:
        font = ImageFont.load_default()

    line_width = max(2, w // 200)
    drawn = 0

    for p in predictions:
        if p.probability < CONF_THRESHOLD:
            continue

        # Convert normalized bbox -> absolute pixels
        left = int(p.bounding_box.left * w)
        top = int(p.bounding_box.top * h)
        width = int(p.bounding_box.width * w)
        height = int(p.bounding_box.height * h)

        # Box
        draw.rectangle([left, top, left + width, top + height], outline="magenta", width=line_width)

        # Label
        label = f"{p.tag_name}: {p.probability * 100:.1f}%"
        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        # Background box with small padding
        pad = 4
        draw.rectangle([left, top - text_h - 2*pad, left + text_w + 2*pad, top], fill="magenta")
        draw.text((left + pad, top - text_h - pad), label, fill="white", font=font)

        drawn += 1

    image.save(out_path, quality=95)
    return drawn

def main():
    # Clear console (optional)
    os.system('cls' if os.name == 'nt' else 'clear')

    # Image from arg or default
    image_file = sys.argv[1] if len(sys.argv) > 1 else "obj-test-images/produce.jpg"
    if not os.path.isfile(image_file):
        print(f"Image not found: {image_file}")
        sys.exit(1)

    try:
        cfg = load_config()
        client = get_client(cfg)
        print(f"Detecting objects in {image_file}...")
        result = detect_image(client, cfg, image_file)

        # Log detected tags over threshold
        kept = [p.tag_name for p in result.predictions if p.probability >= CONF_THRESHOLD]
        if kept:
            print("Detected (≥ {0:.0%}):".format(CONF_THRESHOLD), ", ".join(sorted(set(kept))))
        else:
            print("No detections above threshold.")

        out_path = "obj-test-images/output.jpg"
        count = annotate_image(image_file, result.predictions, out_path)
        print(f"Saved {'annotated' if count else 'original'} image to {out_path} (boxes drawn: {count}).")

    except Exception as ex:
        # Useful, actionable error output
        print("Error:", ex)
        sys.exit(2)

if __name__ == "__main__":
    main()
