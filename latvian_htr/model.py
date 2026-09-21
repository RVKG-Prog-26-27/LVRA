"""
Model architecture - this is the corrected version of the flowchart's "AI" box.

On the diagram, CNN and BiLSTM were drawn as two independent branches off "AI".
For text recognition that's not how it works: they're sequential stages of one
pipeline (a CRNN), plus a CTC layer the diagram was missing entirely:

    image -> CNN (visual features, replaces "Līnijas atpazišana" /
              "Pareiza novietošana" - i.e. stroke patterns + spatial layout)
          -> reshape rows->sequence
          -> BiLSTM x2 (context over the sequence, replaces "Vārdu atpazīšana" /
              "Pieturzīmes" / "Cipari" - i.e. word/punctuation/digit recognition
              needs context from neighboring characters, which only the LSTM
              stage gives you)
          -> Linear classifier over the alphabet
          -> CTC loss/decoding (aligns variable-length predictions to text
              without needing per-character bounding boxes)

CTC is what makes "Teksta zonas atpazīšana????" mostly unnecessary for the
input side too: you don't need to segment characters up front, just feed
whole line images in.
"""

import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """Reduces a [B,1,32,W] line image to a [B,512,1,W'] feature map."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),      # 32x W -> 16 x W/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),    # -> 8 x W/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),  # -> 4 x W/4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),  # -> 2 x W/4
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(inplace=True),  # -> 1 x (W/4 - 1)
        )

    def forward(self, x):
        return self.net(x)


class BiLSTMHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden, num_layers=num_layers, bidirectional=True, batch_first=False
        )
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        # x: [T, B, in_dim]
        out, _ = self.lstm(x)
        return self.fc(out)  # [T, B, num_classes]


class CRNN(nn.Module):
    def __init__(self, num_classes: int, lstm_hidden: int = 256):
        super().__init__()
        self.cnn = CNNBackbone()
        self.rnn = BiLSTMHead(in_dim=512, hidden=lstm_hidden, num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 1, 32, W]
        returns log-probs: [T, B, num_classes], T = sequence length after CNN downsampling
        """
        feats = self.cnn(images)              # [B, 512, 1, W']
        feats = feats.squeeze(2)               # [B, 512, W']
        feats = feats.permute(2, 0, 1)         # [W'(=T), B, 512]
        logits = self.rnn(feats)               # [T, B, num_classes]
        return logits.log_softmax(dim=2)

    def output_length(self, input_width: int) -> int:
        """Sequence length T the CNN produces for a given input image width (for CTC input_lengths)."""
        w = input_width // 2 // 2  # two stride-2 pools
        w = w - 1                  # final kernel=2,stride=1,pad=0 conv
        return max(w, 1)
