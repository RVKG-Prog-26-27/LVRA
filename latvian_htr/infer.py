"""
Inference.

Usage:
    python infer.py --checkpoint checkpoints/crnn_step2000.pt --image samples/sample_0.png

Maps to the flowchart's OUTPUT branch: "Teksts digitālajā veidā" ->
"Koeficientu pārveidošana burtos" (= CTC decode, logits -> characters) ->
"Teksta rakstīšana" (= printing/returning the final string).
"""

import argparse

import torch
from PIL import Image

from alphabet import NUM_CLASSES, decode_greedy
from dataset import image_to_tensor
from model import CRNN
from synthetic_data import TARGET_HEIGHT


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("L")
    w, h = img.size
    new_w = max(1, int(w * (TARGET_HEIGHT / h)))
    img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
    return image_to_tensor(img).unsqueeze(0)  # [1,1,H,W]


def predict(model: CRNN, image_tensor: torch.Tensor, device: torch.device) -> str:
    model.eval()
    with torch.no_grad():
        log_probs = model(image_tensor.to(device))  # [T,1,C]
        pred_indices = log_probs.argmax(dim=2).squeeze(1).tolist()  # [T]
    return decode_greedy(pred_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    image_tensor = load_image(args.image)
    text = predict(model, image_tensor, device)
    print(f"Recognized text: {text!r}")
