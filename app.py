import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from PaintWidget import PaintWidget  # Assuming PaintWidget is in the same directory


def on_confirm():
    result = paint_widget.confirm()
    if result:
        predicted_digit, confidence = result
        result_label.setText(
            f"Predicted digit: {predicted_digit}\nConfidence: {confidence:.2f}"
        )
        result_label.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = QMainWindow()
    main_window.setWindowTitle("Draw a Digit")

    paint_widget = PaintWidget()
    clear_button = QPushButton("Clear")
    clear_button.clicked.connect(paint_widget.clear)
    confirm_button = QPushButton("Confirm")
    confirm_button.clicked.connect(paint_widget.confirm)

    result_label = QMessageBox()
    confirm_button.clicked.connect(on_confirm)

    layout = QVBoxLayout()
    layout.addWidget(paint_widget)
    layout.addWidget(confirm_button)
    layout.addWidget(clear_button)

    container = QWidget()
    container.setLayout(layout)
    main_window.setCentralWidget(container)

    main_window.show()
    sys.exit(app.exec())
