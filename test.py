from models import Generator
from dataset import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from PIL import Image

imagePath = ""
checkpointModel = ""
channels = 3
residualBlocks = 23

os.makedirs("image/output", exist_ok = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(channels, firlters = 64, numResBlocks = residualBlocks).to(device)
generator.load_state_dict(torch.load(checkpointModel))
generator.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normailze(mean, std)])

imageTensor = Variable(transform(Image.open(imagePath))).to(device).unsqueeze(0)

with torch.no_grad():
    srImage = denormalize(generator(imageTensor)).cpu()

fn = imagePath.split("/")[-1]
save_image(srImage, f"images/output/sr-{fn}")