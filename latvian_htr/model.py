"""
Neural network architecture.

The model is called CRNN: C = convolutional, R = recurrent. It is the standard scheme for reading text from images.

1) CNN - turns the image into a sequence of "descriptions", one per narrow vertical slice (left to right).

2) BiLSTM - the "memory and context" part. It reads that sequence in both directions: left to right AND right
   to left. Thanks to that it knows what a letter means next to other letters; for example, an unreadable
   squiggle between "s" and "l" is most likely an "a".

3) Linear + CTC - the "answer". For every slice it chooses which character it is (or "blank"). CTC is a way
   to learn without marking where exactly in the image each letter appears.
"""

import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """
    Shapes:
      B - batch (how many images are processed at once);
      1 - one channel (grayscale image);
      64 - image height in pixels (TARGET_HEIGHT); W - image width (different for every image).
    Output: the height shrinks to 1, the number of "channels" grows to 512, the width is divided by x_stride.
    """

    def __init__(self, x_stride: int = 2):
        super().__init__()
        if x_stride not in (2, 4):
            raise ValueError("x_stride must be 2 or 4")
        # x_stride=4: the 2nd block also halves the width. Then every following (the heaviest) convolution and the
        # BiLSTM work on half the width -> training is about 2x faster. It adds no trainable weights (MaxPool has
        # none), so checkpoints are interchangeable between the two modes, but the network has to be retrained.
        # T = W // 4 is enough for lines up to ~60 characters at height 64.
        pool2 = nn.MaxPool2d(2, 2) if x_stride == 4 else nn.MaxPool2d((2, 1), (2, 1))
        # Layers used below:
        # - Conv2d - convolution; the second number is how many output "channels" it produces (64, 128, 256, 512).
        # - ReLU - replaces all negative numbers by 0 and keeps positive ones. Without such nonlinear layers the
        #   network would collapse into one big multiplication and could not learn anything complicated.
        # - MaxPool2d(2, 2) - "squeeze": keeps only the biggest number of every 2x2 square, so the image becomes
        #   2x smaller in height and in width.
        # - MaxPool2d((2, 1), (2, 1)) - the same, but only the height shrinks. The width is needed because the
        #   text is "read" along it, from left to right.
        # - BatchNorm2d - "normalisation": brings numbers to a convenient scale so training is faster and stabler.
        self.net = nn.Sequential(
            # Block 1: 1 channel -> 64. The only place that always halves the WIDTH: 64 x W -> 32 x W/2
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            # Block 2: 64 -> 128; height only (x_stride=2) or height and width (x_stride=4): -> 16 x W/2 (or W/4)
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), pool2,
            # Block 3: 128 -> 256 -> 256, height only: -> 8
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            # Block 4: 256 -> 512 -> 512 with normalisation, height only: -> 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            # The last convolution has a (4, 1) window: it "eats" the remaining 4 rows of height -> height = 1,
            # the width does NOT change. So T = W // 2 (x_stride=2) or W // 4 (x_stride=4).
            nn.Conv2d(512, 512, (4, 1), 1, 0), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BiLSTMHead(nn.Module):
    """
    The "head" of the network: reads the feature sequence and produces character scores.

    LSTM is a layer with "memory". It reads the data step by step and remembers what happened before, which
    matters for text: to understand a letter you have to see its neighbours. "Bi" means bidirectional: two LSTMs,
    one reads left to right, the other right to left, and their results are glued together. So every position
    knows what came before it and what comes after it.
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 2):
        super().__init__()
        # in_dim - how many numbers describe one slice of the image (512 for us);
        # hidden - size of the LSTM "memory";
        # num_layers=2 - two LSTM layers stacked: the second one reads the result of the first and processes it deeper;
        # bidirectional=True - read in both directions;
        # batch_first=False - order of dimensions: [step, image in batch, features].
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=num_layers, bidirectional=True, batch_first=False)
        # Linear - an ordinary fully connected layer: turns the LSTM numbers into a score for every alphabet
        # symbol. The input is hidden * 2 because both directions are glued together.
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        # x: [T, B, in_dim] - T slices of the image (time steps), B images in the batch, in_dim features each.
        out, _ = self.lstm(x)   # what the LSTM "understood" at every step (the second value, memory, is not needed)
        return self.fc(out)     # [T, B, num_classes]: a score of every symbol for every step


class CRNN(nn.Module):
    """The whole model: "eyes" (CNN) + "understanding" (BiLSTM)."""

    def __init__(self, num_classes: int, lstm_hidden: int = 256, x_stride: int = 2):
        super().__init__()
        self.x_stride = x_stride
        self.cnn = CNNBackbone(x_stride)
        # The head takes 512 features (the number of CNN output channels) and returns num_classes scores
        # (the number of alphabet symbols including blank).
        self.rnn = BiLSTMHead(in_dim=512, hidden=lstm_hidden, num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 1, 64, W] - a batch of text-line images.
        Returns log-probabilities of shape [T, B, num_classes], where T is the sequence length after the CNN
        has squeezed the image width.
        """
        feats = self.cnn(images)              # [B, 512, 1, W']  - features found by the CNN
        feats = feats.squeeze(2)              # [B, 512, W']     - drop the height dimension (it is 1)
        # permute reorders the dimensions: we have [image, features, step], the LSTM wants [step, image, features].
        feats = feats.permute(2, 0, 1)        # [W' (= T), B, 512]
        logits = self.rnn(feats)              # [T, B, num_classes]
        # log_softmax turns scores into log-probabilities (they sum to 100% over the symbols at every step) - the
        # form the CTC loss expects. .float(): under mixed precision softmax and CTC are still computed in float32.
        return logits.float().log_softmax(dim=2)

    def output_length(self, input_width: int) -> int:
        """How many steps T the CNN output has for an image of the given width (CTC needs it).
        x_stride=2: the width is squeezed once (the first MaxPool(2,2)) -> T = W // 2.
        x_stride=4: twice -> T = (W // 2) // 2.
        If you change the CNN (pooling / kernels), THIS FORMULA MUST BE CHANGED TOO, otherwise CTC sees a wrong
        length. (The web server has its own copy of the architecture in web-app/server/htr_numpy.py.)"""
        t = input_width // 2
        if self.x_stride == 4:
            t //= 2
        return max(t, 1)

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        """The same for a whole batch (widths: a long tensor of shape [B])."""
        t = widths // 2
        if self.x_stride == 4:
            t = t // 2
        return t.clamp(min=1)


def save_checkpoint(model: CRNN, path: str, step: int | None = None):
    """Checkpoint = weights + architecture settings (x_stride), so infer/evaluate build the same network."""
    torch.save({"model": model.state_dict(), "x_stride": model.x_stride, "step": step}, path)


def load_checkpoint(path: str, map_location="cpu") -> tuple[dict, int]:
    """Reads both the current format (a dict with 'model') and the old one (a plain state_dict).
    Returns (weights, x_stride)."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt.get("x_stride", 2)
    return ckpt, 2


def check_num_classes(state: dict, num_classes: int, path: str = "checkpoint"):
    """Fail with a clear message if the checkpoint was trained with a different alphabet."""
    have = state["rnn.fc.weight"].shape[0]
    if have != num_classes:
        raise ValueError(
            f"{path} has {have} classes, but alphabet.py defines {num_classes}. The checkpoint was trained with a "
            "different alphabet: retrain the model (or restore the alphabet.py it was trained with)."
        )


def load_model(path: str, num_classes: int, device) -> CRNN:
    """Builds a CRNN with the same architecture as the checkpoint and loads the weights."""
    state, x_stride = load_checkpoint(path, device)
    check_num_classes(state, num_classes, path)
    model = CRNN(num_classes=num_classes, x_stride=x_stride).to(device)
    model.load_state_dict(state)
    return model
