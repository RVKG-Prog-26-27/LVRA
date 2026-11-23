"""
Recognition (inference): take an already trained network and read the text from an image.

This is the last step of the whole chain: "image -> the network outputs numbers -> the numbers are turned into
letters (CTC decoding) -> the finished text is printed".

    python infer.py --checkpoint checkpoints/crnn_best.pt --image line.png
"""

import argparse

import torch

from alphabet import NUM_CLASSES, decode_beam_search
from confidence import WordResult, format_categories, read_words, site_text
from dataset import image_to_tensor, open_line_image
from model import CRNN, load_model


def load_image(path: str) -> torch.Tensor:
    """Opens an image file and prepares it exactly as during training (scaled to TARGET_HEIGHT, contrast
    normalised). Returns a tensor [1, 1, 64, W]: a "batch with one image" - the network always expects a batch,
    even when there is a single image."""
    return image_to_tensor(open_line_image(path)).unsqueeze(0)


def predict(model: CRNN, image_tensor: torch.Tensor, device: torch.device, beam_width: int = 5) -> str:
    """Runs the image through the network and returns the recognised text."""
    model.eval()  # "evaluation" mode (not training): the layers work in the stable mode
    with torch.no_grad():  # do not accumulate information for training - saves memory and time
        log_probs = model(image_tensor.to(device))  # [T, 1, C] - symbol probabilities at every step
        sequence_log_probs = log_probs.squeeze(1)   # drop the batch dimension: [T, 1, C] -> [T, C]
    return decode_beam_search(sequence_log_probs, beam_width=beam_width)   # keeps several best CTC prefixes


def predict_with_words(model: CRNN, image_tensor: torch.Tensor, device: torch.device,
                       beam_width: int = 5) -> tuple[str, list[WordResult]]:
    """Like predict(), but also splits the line into words and gives every word a confidence and a category
    (1 - could not recognise, 2 - not sure, 3 - sure; see confidence.py). The text itself is read exactly as before."""
    model.eval()
    with torch.no_grad():
        sequence_log_probs = model(image_tensor.to(device)).squeeze(1)   # [T, C]
    text = decode_beam_search(sequence_log_probs, beam_width=beam_width)
    return text, read_words(sequence_log_probs)


# Runs only when the file is started directly ("python infer.py ...").
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)  # path to the file with the network weights
    parser.add_argument("--image", type=str, required=True)       # path to the image to read
    parser.add_argument("--beam-width", type=int, default=5, help="beam size for CTC beam search")
    args = parser.parse_args()

    # Use the video card if there is one, otherwise the processor.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load_model understands both checkpoint formats (and x_stride).
    model = load_model(args.checkpoint, NUM_CLASSES, device)

    image_tensor = load_image(args.image)
    text, words = predict_with_words(model, image_tensor, device, beam_width=args.beam_width)
    # !r prints the string "as in code" - in quotes, so spaces at the edges are visible.
    print(f"Text: {text!r}")
    for w in words:
        shown = "-" if w.site_value is None else w.site_value
        print(f"  {w.raw!r:<24} {w.confidence * 100:5.1f}%  {w.category} пункт  -> сайт: {shown}")
    print(f"For the site: {site_text(words)!r}")
    print(format_categories(words))
