import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19Model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19Model.features.children())[:35])

        def forward(self,img):
            return self.vgg19_54(img)

class DenseResidualBlock(nn.Module):

    def __init__(self, filters, resScale = 0.2):
        super(DenseResidualBlock, self).__init__()
        self.resScale = resScale

        def residualBlock(inFeatures, final_layer = False):
            layers = [nn.Conv2d(inFeatures, filters, kernel_size = 3, stride = 1, padding = 1, bias = True)]
            if not final_layer:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.rb1 = residualBlock(inFeatures = 1 * filters)
        self.rb2 = residualBlock(inFeatures = 2 * filters)
        self.rb3 = residualBlock(inFeatures = 3 * filters)
        self.rb4 = residualBlock(inFeatures = 4 * filters)
        self.rb5 = residualBlock(inFeatures = 5 * filters, final_layer = True)
        self.residualBlocks = [self.rb1,self.rb2,self.rb3,self.rb4,self.rb5]
        

    

    def forward(self, x):
        inputs = x
        for block in self.residualBlocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1) # Concatenates tensors (Skipped connections)
        return out.mul(self.resScale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, resScale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.resScale = resScale
        self.denseBlocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.denseBlocks(x).mul(self.resScale) + x

class Generator(nn.Module):
    def __init__(self, channels, filters =64, numResBlocks = 16, numUpsample = 2):
        super(Generator, self).__init__()

        # First Layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size = 3, stride = 1, padding = 1)

        #Residual Blocks
        self.resBlocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range (numResBlocks)])

        # Second Conv Layer Post Residual Blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size = 3, stride = 1, padding = 1)

        #Upsampling Layers
        upsampleLayers = []
        for _ in range(numUpsample):
            upsampleLayers += [
                nn.Conv2d(filters, filters * 4, kernel_size = 3, stride = 1, padding = 1),
                nn. LeakyReLU(),
                nn.PixelShuffle(upscale_factor = 2),
            ]
        self.upsampling = nn.Sequential(*upsampleLayers)

        #Final Output Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size = 3, stride = 1, padding = 1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.resBlocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, inputShape):
        super(Discriminator, self).__init__()

        def discriminatorBlock(inFeatures, outFilters, firstBlock = False):
            layers = []
            layers.append(nn.Conv2d(inFilters, outFilters, kernel_size = 3, stride = 1, padding = 1))
            if not firstBlock:
                layers.append(nn.BatchNorm2d(outFilters))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            layers.append(nn.Conv2d(outFilters, outFilters, kernel_size = 3, stride = 2, padding = 1))
            layers.append(nn.BatchNorm2d(outFilters))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        self.inputShape = inputShape
        inChannels, inHeight, inWidth = self.inputShape
        patchH, patchW = int(inHeight / 2 ** 4), int(inWidth / 2 ** 4)
        self.outputShape = (1, patchH, patchW)

        layers = []
        inFilters = inChannels
        for i, outFilters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminatorBlock(inFilters, outFilters, firstBlock=(i == 0)))
            inFilters = outFilters

        layers.append(nn.Conv2d(outFilters, 1, kernel_size = 3, stride = 1, padding = 1))

        self.model = nn.Sequential(*layers)

    

    def forward(self, img):
        return self.model(img)


