import sys
import numpy as np
import matplotlib.pyplot as plt
from qtpy.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from qtpy.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2

class VideoPlayerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.frames = self.read_frames()

        self.initUI()

    def read_frames(self):
        cap = cv2.VideoCapture('20220721_H36_E2.mp4')
        video_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        return video_frames

    def initUI(self):
        self.setWindowTitle('Video Player')
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # Create a Matplotlib figure and add it to the layout
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(np.zeros_like(self.frames[0]))  # Placeholder image
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Add play button
        self.play_button = QPushButton('Play', self)
        self.play_button.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_button)

        self.current_frame = 0
        self.is_playing = False

        # Set up a timer for video playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video)
        self.timer_interval = 1  # Milliseconds between each frame update

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.setText('Pause')
            self.timer.start(self.timer_interval)
        else:
            self.play_button.setText('Play')
            self.timer.stop()

    def update_video(self):
        if self.current_frame < len(self.frames):
            frame = self.frames[self.current_frame]
            self.im.set_data(frame)
            self.canvas.draw()
            self.current_frame += 1
        else:
            self.current_frame = 0
            self.toggle_playback()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayerApp()
    player.show()
    sys.exit(app.exec_())