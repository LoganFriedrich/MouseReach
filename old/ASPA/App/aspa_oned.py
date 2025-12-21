# Author: Sunil Mathew
# 2023
# Spinal cord injury research (Animal model) at Blackmore Lab, Marquette University
# Dr. Murray Blackmore, Logan Friedrich, Dr. Sunil Mathew
from collections import deque
import traceback
import os
import sys
import cv2

from qtpy import QtCore, QtGui, QtWidgets, uic
from qtpy.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QStyle
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QImage, QPixmap

core_path = os.path.join(os.path.dirname(__file__), 'core')
sys.path.append(core_path)

from frame_processor import FrameInfoProcessor
from progressbar import Progressbar

class MainWindow(QMainWindow):

    #region Initialization

    def __init__(self):
        super().__init__()
        uic.loadUi('App/aspa.ui', self)

        self.addFileMenu()
        
        self.init_ui()
        self.load_data()

    def resizeEvent(self, event):
        """
        To draw the progress bar with right size
        """
        try:
            QtWidgets.QMainWindow.resizeEvent(self, event)

        except:
            print("resizeEvent")
            print(traceback.format_exc())

    def init_ui(self):
        widget = QWidget()
        widget.setLayout(self.vLytMainWin)
        self.setCentralWidget(widget)

        # self.init_progressbar()
        self.init_video_list()
        self.init_frameviewer()
        self.init_stats()

    #endregion Initialization

    #region Video List

    def init_video_list(self):
        try:
            video_list = []
            self.video_path_list = []
            folder_path = os.path.join('Shared Data','Single_Animal')
            if not os.path.exists(folder_path):
                print(f'{folder_path} not found')
                return

            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.mp4'):
                        video_list.append(file)
                        self.video_path_list.append(os.path.join(root, file))

            # Remove duplicates
            video_list = list(set(video_list))
            self.video_path_list = list(set(self.video_path_list))
            # Sort video list
            video_list.sort(key=lambda x: ((int(x.split('_')[0]), x.split('_')[1], x.split('_')[2])))      

            self.cmbVideoList.addItems(video_list)

            self.cmbVideoList.currentIndexChanged.connect(self.cmbVideoList_currentIndexChanged)
            self.cmbVideoList.setCurrentIndex(0)

        except:
            print(traceback.format_exc())

    def cmbVideoList_currentIndexChanged(self, index):
        try:
            self.load_data()

        except:
            print(traceback.format_exc())

    #endregion Video List

    #region Load data

    def load_data(self):
        file = self.cmbVideoList.currentText()
        file_path = ''
        for file_path in self.video_path_list:
            if file in file_path:
                break
        
        csv_file = file_path.replace('.mp4', 'DLC_resnet50_MSA - V1Mar3shuffle1_1030000.csv')
        self.frame_proc.frame_info.read_csv(csv_file)
        self.frame_proc.vdo_player.set_df(self.frame_proc.frame_info.df)
        self.frame_proc.stats.df = self.frame_proc.frame_info.df
        self.btnDetectSwipes_clicked()
        # self.frame_proc.stats.show_stats()

        self.frame_proc.vdo_player.read_video(video_path=file_path)
        # self.init_video_player(file_path)

    def init_video_player(self, vdo_path):
        # Create a video buffer
        buffer_size = 200
        self.frame_buffer = deque([] * buffer_size)
        # Create a timer
        self.timer = QTimer(self)

        # Create the video capture thread
        self.cap = cv2.VideoCapture(vdo_path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        # Set thumbnails
        self.set_thumbnails()
      
        # Connect timer to update_frame method
        self.timer.timeout.connect(self.update_frame)  
        self.update_frame()

        # Play button
        self.btnPlay.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btnPlay.clicked.connect(self.play_video)

        # Slider
        self.sldrFrames.setMinimum(0)
        self.sldrFrames.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        # self.sldrFrames.valueChanged.connect(self.on_slider_value_changed)

        self.paused = True

    #endregion Load data

    #region File Menu

    def addFileMenu(self):
        self.dirpath = QtCore.QDir.currentPath()
        self.filter_name = 'All files (*.*)'
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')

        # Create new action
        openAction = QtGui.QAction(QtGui.QIcon('open.png'), '&Open', self)        
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open nii file')
        openAction.triggered.connect(self.openCall)

        fileMenu.addAction(openAction)

        return fileMenu

    def openCall(self, index):
        # the signal passes the index of the clicked item
        self.FILE = QtWidgets.QFileDialog.getOpenFileName(self, caption='Choose File',
                                                    directory=self.dirpath,
                                                    filter=self.filter_name)[0]

        if '' != self.FILE:
            self.process_frames(self.FILE)

    def openFolderCall(self, index):
        # the signal passes the index of the clicked item
        self.Files = QtWidgets.QFileDialog.getOpenFileNames(self, caption='Choose File',
                                                    directory=self.dirpath,
                                                    filter=self.filter_name)[0]

        if self.Files:
            self.currVolIdx = 0
            self.updateVolumeViewer(self.Files[0])

            self.updateMplViewer(self.Files[0])

            # self.trainPGM(self.Files)
    
    #endregion File Menu

    #region Progressbar

    def init_progressbar(self):
        try:
            self.pb_training = Progressbar()
            self.vLytProgbar.addWidget(self.pb_training.can_pb)
            self.pb_training.can_pb.draw()

        except:
            print(traceback.format_exc())

    #endregion Progressbar

    #region Frameviewer

    def init_frameviewer(self):
        try:
            self.tabFrames.setLayout(self.vLytFrameViewer)
            self.tabFrameInfo.setLayout(self.vLytDLCTable)
            self.tabSwipeInfo.setLayout(self.vLytSwipeInfo)
            self.tabStats.setLayout(self.vLytStats)

            self.frame_proc = FrameInfoProcessor(tblFrameInfo=self.tblFrameInfo, 
                                                 tblSwipeInfo=self.tblSwipeInfo, 
                                                 reachFilter=self.reachOutcomeFilter,
                                                 ui=True)
            self.frameVideoPlayer.hide()
            self.frameMPLVideoPlayer.setLayout(self.vLytMPLVideo)
            self.vLytMPLVideo.addWidget(self.frame_proc.vdo_player.can_vp)
            # self.vLytHistogram.addWidget(self.frame_proc.stats.can_stats)
            self.vLytRightHandY.addWidget(self.frame_proc.action_detector.can_feat)

            # self.spbPlaybackSpeed.valueChanged.connect(self.spbPlaybackSpeed_valueChanged)
            self.chkMarkFrames.stateChanged.connect(self.chkMarkFrames_stateChanged)
            self.chkSwipes.stateChanged.connect(self.chkSwipes_stateChanged)
            self.btnDetectSwipes.clicked.connect(self.btnDetectSwipes_clicked)
            # self.btnDetectTrayMotion.clicked.connect(self.btnDetectTrayMotion_clicked)

        except:
            print(traceback.format_exc())


    def chkMarkFrames_stateChanged(self, event):    
        try:
            if self.chkMarkFrames.isChecked():
                self.frame_proc.vdo_player.mark_frames = True
            else:
                self.frame_proc.vdo_player.mark_frames = False

        except:
            print(traceback.format_exc())

    def btnDetectTrayMotion_clicked(self):
        try:
            self.frame_proc.detect_tray_motion(start=self.spbFrameStart.value(), end=self.spbFrameEnd.value(),
                                               std_frames=self.spbSwipeInterval.value(), std_thresh=self.spbValleyInterval.value())
        except:
            print(traceback.format_exc())

    def btnDetectSwipes_clicked(self):
        try:
            summary = self.frame_proc.detect_swipes(start=self.spbFrameStart.value(), end=self.spbFrameEnd.value(),
                                          threshold=self.spbSwipeThreshold.value(), rh_y_conf=self.dspbRHConf.value(),
                                          std_motion=self.spbStdMotion.value(),
                                          title=self.frame_proc.get_info(self.cmbVideoList.currentText()))
            
            print(summary)
        except:
            print(traceback.format_exc())

    def chkSwipes_stateChanged(self, event):
        try:
            if self.chkSwipes.isChecked():
                self.frame_proc.vdo_player.play_swipes_only = True
            else:
                self.frame_proc.vdo_player.play_swipes_only = False
        except:
            print(traceback.format_exc())

    def spbFrames_valueChanged(self, value):
        try:
            self.frame_proc.vdo_player.current_frame_idx = value
            self.frame_proc.vdo_player.update_thumbnails()
        except:
            print(traceback.format_exc())

    def spbPlaybackSpeed_valueChanged(self, value):
        try:
            self.frame_proc.vdo_player.set_playback_speed(value)
        except:
            print(traceback.format_exc())

    #endregion Frameviewer

    #region Video Player

    def play_video(self):
        self.paused = not self.paused
        if self.paused:
            self.timer.stop()
            self.btnPlay.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)
            )
        else:
            self.btnPlay.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause)
            )
            self.timer.start()


    # Set thumbnails 
    def set_thumbnails(self):

        # Capture frames for thumbnails
        for i in range(7):
            ret, frame = self.cap.read()
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize image to thumbnail size
            h, w, ch = rgb_frame.shape
            new_width = int(w / 4)
            new_height = int(new_width * (h / w)) 
            rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap(qimg)

            label = QLabel()
            label.setPixmap(pixmap)
            self.hLytThumbnails.addWidget(label)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update progress bar
            self.sldrFrames.setValue(int(100 * self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
         
            # Convert ndarray to QImage
            height, width, channel = rgb_frame.shape
            bytesPerLine = 3 * width
            q_image = QImage(rgb_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)

            # Add to buffer
            # self.frame_buffer.append(q_image)

            # Convert QImage to QPixmap
            pixmap = QPixmap(q_image)
            
            # Display image
            self.lblFrameView.setPixmap(pixmap)

            del frame, rgb_frame, q_image, pixmap

    def on_slider_value_changed(self, value):
        print(f'slider: {value}')
        # Calculate position to seek to
        pos = int(value / 100 * self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)  

        # Update widget 
        self.update_frame()

    #endregion Video Player

    #region Process Frames

    def process_frames(self, file_path):
        try:
            # Check if mp4 file
            if file_path.endswith('.mp4'):
                self.frame_proc.vdo_player.read_video_thread(file_path)
                # self.read_csv_thread(file_path)
        except:
            print(traceback.format_exc())


    #endregion Process Frames

    #region Stats

    def init_stats(self):
        try:
            self.filter = {}
            self.frame_proc.stats.tblSummary = self.tblSummary
            self.frame_proc.stats.display_df()
            self.cmbTestDate.addItems(['All'] + list(self.frame_proc.stats.dates))
            self.cmbTestDate.currentIndexChanged.connect(self.filter_summary_table)
            self.cmbTestType.addItems(['All', 'Train', 'Post-Injury', 'Rehab'])
            self.cmbTestType.currentIndexChanged.connect(self.filter_summary_table)

            self.cmbTrayType.addItems(['All', 'Pillar', 'Easy', 'Flat'])
            self.cmbTrayType.currentIndexChanged.connect(self.filter_summary_table)

            self.cmbAnimalGroup.addItems(['H', 'D'])
            self.cmbAnimalGroup.currentIndexChanged.connect(self.filter_summary_table)

            # Animal ids upto 40
            self.cmbAnimalId.addItems(['All'] + ['0'+str(i) if i<10 else str(i) for i in range(1, 41)])
            self.cmbAnimalId.currentIndexChanged.connect(self.filter_summary_table)

            self.cmbTrayId.addItems(['All', '1', '2', '3', '4'])
            self.cmbTrayId.currentIndexChanged.connect(self.filter_summary_table)

            self.tabSummaryTable.setLayout(self.vLytSummaryTable)
            self.tabPlots.setLayout(self.vLytPlots)
            # self.cmbSwipeFeature.addItems(['Swipe speed', 'Swipe duration', 'Swipe area', 'Attention', 'Total swipes'])
            self.cmbSwipeFeature.addItems(self.frame_proc.stats.plot_df.columns.tolist())
            self.cmbSwipeFeature.currentIndexChanged.connect(self.plot_swipe_feature)

            self.cmbPlotGroup.addItems(['Animal ID', 'Date', 'Pellets'])
            self.cmbPlotGroup.currentIndexChanged.connect(self.plot_swipe_feature)

            self.cmbPlotType.addItems(['Box', 'Scatter', 'Line'])
            self.cmbPlotType.currentIndexChanged.connect(self.plot_swipe_feature)

            self.vLytPlot.addWidget(self.frame_proc.stats.can_stats)
        except:
            print(traceback.format_exc())

    def plot_swipe_feature(self, index):
        try:
            self.filter_summary_table()
            
        except:
            print(traceback.format_exc())

    def filter_summary_table(self, index=None):
        try:
            self.filter['Test Date'] = self.cmbTestDate.currentText() if self.cmbTestDate.currentText() != 'All' else ''

            self.filter['Test Type'] = self.cmbTestType.currentText() if self.cmbTestType.currentText() != 'All' else ''
                
            self.filter['Tray Type'] = self.cmbTrayType.currentText()if self.cmbTrayType.currentText() != 'All' else ''
                
            self.filter['Group'] = self.cmbAnimalGroup.currentText() if self.cmbAnimalGroup.currentText() != 'All' else ''
                
            self.filter['Animal #'] = self.cmbAnimalId.currentText() if self.cmbAnimalId.currentText() != 'All' else ''
                
            self.filter['Tray #'] = self.cmbTrayId.currentText() if self.cmbTrayId.currentText() != 'All' else ''
                
            if index is not None:
                self.frame_proc.stats.filter_data(filter=self.filter)

            self.frame_proc.stats.plot_swipe_feature(feature=self.cmbSwipeFeature.currentText(), 
                                                     plot_type=self.cmbPlotType.currentText(),
                                                     plot_group=self.cmbPlotGroup.currentText(),
                                                     filter=self.filter)
        except:
            print(traceback.format_exc())

    #endregion Stats

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()



