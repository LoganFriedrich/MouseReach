from ast import main
import cv2
from PyQt5.Qt import QStyle
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QHBoxLayout, QWidget, QVBoxLayout, QProgressBar
from PyQt5.QtCore import Qt, QTimer
import sys

class VideoWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt5 Video") 
        # self.disply_width = 640
        # self.display_height = 480
        
        # Create the label that holds the image
        self.image_label = QLabel(self)
        # self.image_label.resize(self.disply_width, self.display_height)

        # Create progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setValue(0)
        self.progressBar.setFormat('%p%')
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setStyleSheet('QProgressBar {border: 2px solid grey; border-radius: 5px; padding: 1px}'
                                        'QProgressBar::chunk {background-color: #CD96CD; width: 10px;}')
        self.progressBar.setFixedHeight(20)
        self.progressBar.setTextVisible(True)

        # Create a timer
        self.timer = QTimer(self)

        # Create the video capture thread
        self.cap = cv2.VideoCapture('20220721_H36_E2.mp4')

        # Set thumbnails
        self.set_thumbnails()
      
        # Connect timer to update_frame method
        self.timer.timeout.connect(self.update_frame)  
        self.update_frame()

        # Start timer 
        # self.timer.start()  

        # Play button
        self.btn_play = QPushButton()
        # self.btn_play.setEnabled(False)
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.clicked.connect(self.play_video)
        
        # Create hbox layout
        controls = QHBoxLayout()     
        controls.addWidget(self.btn_play)

        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.progressBar)
        main_layout.addLayout(controls)
        main_layout.addLayout(self.thumbnail_layout)
        
        self.setLayout(main_layout)

        self.paused = True

    def resizeEvent(self, event):
        self.image_label.resize(self.width(), self.height())

    def play_video(self):
        self.paused = not self.paused
        if self.paused:
            self.timer.stop()
            self.btn_play.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)
            )
        else:
            self.btn_play.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause)
            )
            self.timer.start()


    # Set thumbnails 
    def set_thumbnails(self):
        # Create thumbnail layout 
        self.thumbnail_layout = QHBoxLayout()

        # Capture frames for thumbnails
        for i in range(5):
            ret, frame = self.cap.read()
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize image to thumbnail size
            rgb_frame = cv2.resize(rgb_frame, (int(rgb_frame.shape[1]/6), int(rgb_frame.shape[0]/4)))
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap(qimg)

            label = QLabel()
            label.setPixmap(pixmap)
            self.thumbnail_layout.addWidget(label)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update progress bar
            self.progressBar.setValue(int(100 * self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
         
            # Convert ndarray to QImage
            height, width, channel = rgb_frame.shape
            bytesPerLine = 3 * width
            q_image = QImage(rgb_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)

            # Convert QImage to QPixmap
            pixmap = QPixmap(q_image)
            
            # Display image
            self.image_label.setPixmap(pixmap)
       
    def on_progress_changed(self, value):
        # Calculate position to seek to
        pos = int(value / 100 * self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)  

        # Update widget 
        self.update_frame()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec())