import os, time, re
import torch
from PIL import Image
from dotenv import load_dotenv

from transformers import (
    AutoProcessor,
    VisionEncoderDecoderModel
)


# -----------------------------
# CONFIG
# -----------------------------

# load_dotenv()
# HF_MODEL = os.environ.get("HF_MODEL")
# LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH")

HF_MODEL = "hoang-quoc-trung/sumen-base"
LOCAL_MODEL_PATH = "./models/sumen-base"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

if not os.path.exists(LOCAL_MODEL_PATH):

    print("Local model not found")
    print("Downloading model from Hugging Face...")

    model = VisionEncoderDecoderModel.from_pretrained(HF_MODEL)
    processor = AutoProcessor.from_pretrained(HF_MODEL)

    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

    model.save_pretrained(LOCAL_MODEL_PATH)
    processor.save_pretrained(LOCAL_MODEL_PATH)

    print("Model downloaded and saved locally")

print("Loading model...")

model = VisionEncoderDecoderModel.from_pretrained(
    LOCAL_MODEL_PATH
).to(device)

processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_PATH
)

print("Model loaded successfully")


# -----------------------------
# INFERENCE FUNCTION
# -----------------------------

def run_inference(image_path):
    print("running inference")

    image = Image.open(image_path).convert("RGB")

    pixel_values = processor.image_processor(
        image,
        return_tensors="pt"
    ).pixel_values

    task_prompt = processor.tokenizer.bos_token

    decoder_input_ids = processor.tokenizer(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids

    with torch.no_grad():

        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.generation_config.max_length,
            num_beams=4,
        )

    sequence = processor.tokenizer.batch_decode(outputs)[0]

    sequence = sequence.replace(
        processor.tokenizer.eos_token, ""
    ).replace(
        processor.tokenizer.pad_token, ""
    ).replace(
        processor.tokenizer.bos_token, ""
    )

    return sequence.strip()

def normalize_latex(text):

    text = text.lower()

    # remove spaces/newlines
    text = re.sub(r"\s+", "", text)

    # remove wrappers
    remove_tokens = [
        "\\left",
        "\\right",
        "$",
        "\\displaystyle",
        "\\!",
        "\\;",
        "\\,",
        "\\:",
    ]

    for token in remove_tokens:
        text = text.replace(token, "")

    return text.strip()

def exact_match(preds, refs):

  correct = 0

  for p, r in zip(preds, refs):

      if p == r:
          correct += 1

  return correct / len(refs)

if __name__ == "__main__":
    print("Running inference...")
    print("================")
    actual = "\\mathrm { T r } \; \gamma ^ { 0 } R _ { 1 / 2 , n } ( \omega ) \propto \omega R _ { n } ( \omega ) \ge 0 \qquad \mathrm { f o r ~ \ o m e g a \ge 0 ~ } ,"
    print("Actual:", normalize_latex(actual))
    stt = time.time()
    result = run_inference("dataset/download.png")
    ett = time.time()
    print("Predicted:", normalize_latex(result))
    print(f"Time taken: {ett - stt} seconds")
    print(f"Exact match: {exact_match(normalize_latex(actual), normalize_latex(result))}")