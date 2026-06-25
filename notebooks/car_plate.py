import torch
import torch.nn as nn

ALPHABET = "0123456789ABCEHIKMOPTXYZ_"
BLANK_IDX = 0  
NUM_CLASSES = len(ALPHABET) + 1  

char_to_idx = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 1..N
idx_to_char = {i + 1: c for i, c in enumerate(ALPHABET)}

FALLBACK_CHAR = "_"
FALLBACK_IDX = char_to_idx[FALLBACK_CHAR]


class CRNN(nn.Module):

    def __init__(self, img_height=128, num_channels=3, hidden_size=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  

            nn.Conv2d(512, 512, 2), 
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.hidden_size = hidden_size

        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, img_height, img_height)  
            conv_out = self.cnn(dummy)  
            _, c, h, w = conv_out.size()
            self.conv_out_c = c
            self.conv_out_h = h
            self.conv_out_w = w
            rnn_input_size = c * h  

        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False
        )

        self.fc = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, x):
        conv = self.cnn(x)  
        B, C, H, W = conv.size()
        conv = conv.permute(3, 0, 1, 2) 
        conv = conv.contiguous().view(W, B, C * H) 
        rnn_out, _ = self.rnn(conv)  
        logits = self.fc(rnn_out)  

        return logits


def encode_plate(text: str) -> torch.Tensor:
    text = text.strip().upper()
    ids = [char_to_idx.get(c, FALLBACK_IDX) for c in text]
    return torch.tensor(ids, dtype=torch.long)


def decode_plate(indices):
    chars = []
    for idx in indices:
        if idx == BLANK_IDX:
            continue
        chars.append(idx_to_char.get(int(idx), FALLBACK_CHAR))
    return "".join(chars)