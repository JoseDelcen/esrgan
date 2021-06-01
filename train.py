
import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataset import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

inEpoch = 0
inNumEpochs = 200
inDatasetName = "img_align_celeba"
inBatchSize = 16
inLearningRate = 0.0002
inB1 = 0.9
inB2 = 0.999
inDecayEpoch = 100
inNumCpu = 8
inHrHeight = 256
inHrWidth = 256
inChannels = 3
inSampleInterval = 100
inCheckpointInterval = 5000
inResidualBlocks = 23
inWarmupBatches = 500
inLambdaAdv = 5e-3
inLambdaPixel = 1e-2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hrShape = (inHrHeight, inHrWidth)

generator = Generator(inChannels, filters = 64, numResBlocks= inResidualBlocks).to(device)
discriminator = Discriminator(inputShape=(inChannels, *hrShape)).to(device)
featureExtractor = FeatureExtractor().to(device)

featureExtractor.eval()

#Losses
criterionGAN = torch.nn.BCEWithLogitsLoss().to(device)
criterioncontent = torch.nn.L1Loss().to(device)
criterionPixel = torch.nn.L1Loss().to(device)

# Load pretrained models
if inEpoch != 0:
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % inEpoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % inEpoch))

optimizerG = torch.optim.Adam(generator.parameters(), lr = inLearningRate, betas=(inB1,inB2))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=inLearningRate, betas=(inB1,inB2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

dataLoader = DataLoader(
    ImageDataset("data/", hrShape = hrShape),
    batch_size = inBatchSize,
    shuffle = True,
    num_workers = inNumCpu
)

# Training

for epoch in range(inEpoch, inNumEpochs):
    for i, imgs in enumerate(dataLoader):

        batchesDone = epoch * len(dataLoader) + 1

        imgsLr = Variable(imgs["lr"].type(Tensor))
        imgsHr = Variable(imgs["hr"].type(Tensor))

        valid = Variable(Tensor(np.ones((imgsLr.size(0), *discriminator.outputShape))), requires_grad = False)
        fake = Variable(Tensor(np.zeros((imgsLr.size(0), *discriminator.outputShape))), requires_grad = False)


        # Train Generators
        optimizerG.zero_grad()

        genHr = generator(imgsLr)

        lossPixel = criterionPixel(genHr, imgsHr)

        if batchesDone < inWarmupBatches:
            lossPixel.backward()
            optimizerG.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, inNumEpochs, i, len(dataLoader), lossPixel.item())
            )
            continue
            
        predReal = discriminator(imgsHr).detach()
        predFake = discriminator(genHr)

        lossGAN = criterionGAN(predFake - predReal.mean(0, keepdim=True), valid)

        genFeatures = featureExtractor(genHr)
        realFeatures = featureExtractor(imgsHr).detach()
        lossContent = criterionContent(genFeatures, realFeatures)

        lossG = lossContent + inLambdaAdv * lossGAN + inLambdaPixel*lossPixel

        lossG.backward()
        optimizerG.step()

        #Train discriminator
        optimizerD.zero_grad()

        predReal = discriminator(imgsHr)
        predFake = discriminator(genHr.detach())

        lossReal = criterionGAN(predReal - predFake.mean(0, keepdim=True), valid)
        lossFake = criterionGAN(predFake - predReal.mean(0, keepdim=True), fake)

        lossD = (lossReal + lossFake) / 2
        
        lossD.backward()
        optimizerD.step()

        #Log Progress

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
            %(
                epoch,
                inNumEpochs,
                i,
                len(dataLoader),
                lossD.item(),
                lossG.item(),
                lossContent.item(),
                lossGAN.item(),
                lossPixel.item(),
            )
        )

        if batchesDone % inSampleInterval == 0:
            imgsLr = nn.fucntional.interpolate(imgsLr, scale_factor = 4)
            imgGrid = denormalize(torch.cat((imgsLr, genHr),-1))
            save_image(imgGrid, "images/training/%d.png" % batchesDone, nrow=1, normalize = False)

        if batchesDone % inCheckpointInterval == 0:
            torch.save(generator.state_dict(), "saved_models/generator_%d.pth" %epoch)
            torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" %epoch)

