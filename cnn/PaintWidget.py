# PaintWidget.py
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QPaintEvent, QPen, QMouseEvent, QPixmap
from PySide6.QtCore import Qt, QPoint
from cnn.model import NeuralNetwork
import torch
from torchvision.transforms.functional import to_pil_image


class PaintWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StaticContents)
        self.setFixedSize(280, 280)  # Same as MNIST image x10 for visibility
        self.canvas = QPixmap(self.size())
        self.canvas.fill(Qt.white)

        self.last_point = QPoint()
        self.drawing = False

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.canvas)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.last_point = event.position().toPoint()
            self.drawing = True

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing and event.buttons() & Qt.LeftButton:
            painter = QPainter(self.canvas)
            pen = QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear(self):
        self.canvas.fill(Qt.white)
        self.update()

    def confirm(self):
        raw_image = self.get_image()
        image = NeuralNetwork.prepare_image(raw_image)
        model = NeuralNetwork()
        model.load_state_dict(
            torch.load("mnist_model.pth", map_location=torch.device("cpu"))
        )
        model.eval()
        with torch.no_grad():
            output = model(image)
            predicted = output.argmax(dim=1, keepdim=True)
            print(f"Predicted digit: {predicted.item()}")
            confidence_output = torch.exp(output)
            confidence = confidence_output.max(dim=1).values.item()
            print(f"Confidence: {confidence:.2f}")
            return predicted.item(), confidence

    def get_image(self):
        """
        Returns the current drawing as a QImage (you can convert to NumPy or save).
        """
        return self.canvas.toImage()
