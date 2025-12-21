import cv2
from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.Qt import QStyle
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QPushButton, QGridLayout, QLabel, QSlider, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
import sys

class VideoPlayer(QWidget):

    def __init__(self, video_path):
        super().__init__()

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        video_widget = QVideoWidget()

        open_button = QPushButton('Open Video')
        open_button.clicked.connect(lambda: self.open_file(video_path))  

        self.play_button = QPushButton()
        self.play_button.setEnabled(False)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_video)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)

        self.status_label = QLabel()
        self.status_label.setText('No video loaded')

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addWidget(open_button)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.position_slider)

        main_layout = QVBoxLayout()
        main_layout.addWidget(video_widget)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)

        self.media_player.setVideoOutput(video_widget)
        
        self.media_player.stateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)

        self.thumbnail_label = QLabel()
        self.thumbnail_timer = QTimer(self, timeout=self.update_thumbnail)

    def open_file(self, video_path):
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.play_button.setEnabled(True)
        self.status_label.setText('Opened {}'.format(video_path))

    def play_video(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()
            self.thumbnail_timer.start(1000)

    def media_state_changed(self, state):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.play_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause)
            )
        else:
            self.play_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)
            )

    def position_changed(self, position):
        self.position_slider.setValue(position)

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def set_position(self, position):
        self.media_player.setPosition(position)

    def update_thumbnail(self):
        vidcap = cv2.VideoCapture(video_path) 
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            step = channel * width
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.thumbnail_label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_path = '20220721_H36_E2.mp4' # set video path here
    player = VideoPlayer(video_path)
    player.show()
    sys.exit(app.exec_())