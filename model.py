import torch.nn as nn
from PySide6.QtGui import QImage
import torch
import PIL.Image as Image
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(28 * 28, 512), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(512, 10), nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    @staticmethod
    def qimage_to_pil(qimage: QImage) -> Image.Image:
        """
        Converts a QImage to a PIL Image.
        """
        qimage = qimage.convertToFormat(QImage.Format_RGBA8888)
        width, height = qimage.width(), qimage.height()
        ptr = qimage.bits()
        arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))
        return Image.fromarray(arr, mode="RGBA")

    @staticmethod
    def prepare_image(qimage: QImage) -> torch.Tensor:
        """
        Prepares the image for prediction by converting it to grayscale,
        resizing it to 28x28 pixels, and normalizing the pixel values.
        """
        pil_image = NeuralNetwork.qimage_to_pil(qimage).convert("L").resize((28, 28))

        tensor = torch.tensor(list(pil_image.getdata()), dtype=torch.float32).view(
            1, 1, 28, 28
        )

        tensor /= 255.0  # Normalize to [0, 1]
        tensor = 1 - tensor  # Invert colors (MNIST is white on black)
        return tensor
