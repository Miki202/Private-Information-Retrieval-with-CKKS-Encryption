import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import AutoModel


model = AutoModel.from_pretrained(
    "quebeccyb/vehitv",
    trust_remote_code=True
).to("cuda" if torch.cuda.is_available() else "cpu")

model.eval()


class Letterbox:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        nw, nh = int(w * scale), int(h * scale)

        img = img.resize((nw, nh), Image.BILINEAR)

        new_img = Image.new("RGB", (self.size, self.size), (self.fill,)*3)
        new_img.paste(img, ((self.size - nw)//2, (self.size - nh)//2))

        return new_img


transform = transforms.Compose([
    Letterbox(256),
    transforms.ToTensor()
])

img = Image.open("example.png").convert("RGB")
x = transform(img).unsqueeze(0).to(next(model.parameters()).device)

with torch.no_grad():
    out = model(x)
    recon = out


plt.subplot(1,2,1)
plt.title("input")
plt.imshow(x.squeeze().cpu().permute(1,2,0))

plt.subplot(1,2,2)
plt.title("reconstruction")
plt.imshow(recon.squeeze().cpu().permute(1,2,0))

plt.show()