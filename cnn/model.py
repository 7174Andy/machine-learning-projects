import torch.nn as nn
import torch.nn.functional as F
from PySide6.QtGui import QImage
import torch
import PIL.Image as Image
import numpy as np
from torchvision.transforms.functional import to_pil_image


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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
