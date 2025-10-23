# Handwritten Digit Recognition using a CNN

This project is a desktop application that recognizes handwritten digits drawn by the user. It uses a Convolutional Neural Network (CNN) built with PyTorch and a GUI created with PySide6.

## Explanation

The project consists of two main parts:

1.  **Model Training (`handwritten-recognition.ipynb`):** A Jupyter notebook that details the process of building, training, and evaluating a CNN on the MNIST dataset. The trained model weights are saved to `mnist_model.pth`.
2.  **GUI Application (`app.py`):** A graphical user interface where you can draw a digit on a canvas. When you click "Confirm", the application processes the drawing, feeds it to the trained CNN model, and displays the predicted digit along with the model's confidence level.

The core components are:
- `model.py`: Defines the `NeuralNetwork` class for the CNN architecture.
- `PaintWidget.py`: A custom PySide6 widget that provides the drawing canvas.
- `app.py`: The main application that brings everything together.

## Tech Stack

- **Python 3.12**
- **PyTorch:** For building and training the neural network.
- **PySide6:** For the graphical user interface.
- **NumPy:** For numerical operations.
- **Matplotlib:** Used in the notebook for data visualization.

## Installation Guide

1.  Ensure you have Python 3.12 and [uv](https://github.com/astral-sh/uv) installed.

2.  Install the dependencies:
    ```bash
    uv sync
    ```

## How to Run

To run the digit recognition application, execute the `app.py` script from within the `cnn` directory:

```bash
uv run app.py
```

To explore the model training process, you can run the Jupyter notebook:

```bash
jupyter notebook handwritten-recognition.ipynb
```
