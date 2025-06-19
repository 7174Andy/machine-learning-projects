import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from PaintWidget import PaintWidget  # Assuming PaintWidget is in the same directory

if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = QMainWindow()
    main_window.setWindowTitle("Draw a Digit")

    paint_widget = PaintWidget()
    clear_button = QPushButton("Clear")
    clear_button.clicked.connect(paint_widget.clear)

    layout = QVBoxLayout()
    layout.addWidget(paint_widget)
    layout.addWidget(clear_button)

    container = QWidget()
    container.setLayout(layout)
    main_window.setCentralWidget(container)

    main_window.show()
    sys.exit(app.exec())
