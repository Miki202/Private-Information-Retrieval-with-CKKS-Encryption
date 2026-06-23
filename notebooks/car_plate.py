import torch
import torch.nn as nn

# Alphabet and encoding
ALPHABET = "0123456789ABCEHIKMOPTXYZ_"
BLANK_IDX = 0  # CTC blank
NUM_CLASSES = len(ALPHABET) + 1  # + blank

char_to_idx = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 1..N
idx_to_char = {i + 1: c for i, c in enumerate(ALPHABET)}

FALLBACK_CHAR = "_"
FALLBACK_IDX = char_to_idx[FALLBACK_CHAR]


class CRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) model for license plate text recognition.

    Architecture:
    - CNN: Extracts visual features from the input image
    - RNN: Processes the sequence of features
    - CTC: Decodes the sequence into text
    """

    def __init__(self, img_height=128, num_channels=3, hidden_size=256):
        super().__init__()

        # Convolutional layers
        self.cnn = nn.Sequential(
            # 128x128
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x64

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32x32

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 16x32

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8x32

            nn.Conv2d(512, 512, 2),  # 7x31 (approximately)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.hidden_size = hidden_size

        # Calculate RNN input size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, img_height, img_height)  # 128x128
            conv_out = self.cnn(dummy)  # (1, C, H', W')
            _, c, h, w = conv_out.size()
            self.conv_out_c = c
            self.conv_out_h = h
            self.conv_out_w = w
            rnn_input_size = c * h  # we'll unroll along the width


        # Recurrent layers
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (B, C, H, W=128)

        Returns:
            logits: Output tensor (T, B, NUM_CLASSES) - logits for CTC
        """
        conv = self.cnn(x)  # (B, C, H', W')
        B, C, H, W = conv.size()

        # Create sequence along width dimension
        conv = conv.permute(3, 0, 1, 2)  # (W', B, C, H')
        conv = conv.contiguous().view(W, B, C * H)  # (T=W', B, C*H')

        rnn_out, _ = self.rnn(conv)  # (T, B, 2*hidden)
        logits = self.fc(rnn_out)  # (T, B, NUM_CLASSES)

        return logits


def encode_plate(text: str) -> torch.Tensor:
    """
    Encode plate text to tensor of indices for CTC

    Args:
        text: Plate text (e.g., 'AA3165TB')

    Returns:
        Tensor of character indices
    """
    text = text.strip().upper()
    ids = [char_to_idx.get(c, FALLBACK_IDX) for c in text]
    return torch.tensor(ids, dtype=torch.long)


def decode_plate(indices):
    """
    Decode sequence of indices to text (without CTC logic)

    Args:
        indices: List or tensor of character indices

    Returns:
        Decoded text string
    """
    chars = []
    for idx in indices:
        if idx == BLANK_IDX:
            continue
        chars.append(idx_to_char.get(int(idx), FALLBACK_CHAR))
    return "".join(chars)