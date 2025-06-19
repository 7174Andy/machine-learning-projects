import torch.nn as nn
import torch


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
    def prepare_image(image):
        """
        Prepares the image for prediction by converting it to grayscale,
        resizing it to 28x28 pixels, and normalizing the pixel values.
        """
        image = image.convert("L").resize((28, 28))
        image = torch.tensor(list(image.getdata()), dtype=torch.float32).view(
            1, 1, 28, 28
        )
        image /= 255.0  # Normalize pixel values to [0, 1]
        return image
