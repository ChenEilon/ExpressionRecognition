# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demoApp.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#

from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
from PyQt5.QtMultimedia import *
import sys
import qdarkstyle
import numpy as np
import cv2
import dlib
import pickle
import glob
import time

DELTA_THRESHOLD = 2
HAPPY_SONG = "./Bamboleo - Gipsy Kings.mp3" #TODO remove
SAD_SONG = "./Goodbye My Lover - James Blunt.mp3" #TODO remove
PLAYLISTS_PATH = r"./Playlists"
REF_POINTS = [4, 14, 18, 20, 22, 23, 25, 27, 28, 31, 32, 36, 37, 38, 40, 42, 43, 45, 46, 47, 49, 51, 52, 53, 61, 63, 65, 67]
EMOTIONS = ["neutral",  "happy", "sadness", "surprise",  "fear", "disgust", "anger"]
MOOD_PREDICTOR_FILENAME = "modelLF.dat"

wanted_landmarks = [i-1 for i in REF_POINTS]

STRING_TITLE = "Emotion recognition music player"
STRING_PLAY = "Play"
STRING_PAUSE = "Pause"
STRING_TRAIN = "Train"
STRING_TRAINING = "Training..."
STRING_SHOW_SELF = "Show Self"
STRING_LABEL = "DEMO APP - Emotion recognition music player"

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        #####Qt Designer part#####
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(360, 360)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.controlLayout = QtWidgets.QWidget(self.centralwidget)
        self.controlLayout.setGeometry(QtCore.QRect(10, 283, 330, 23))
        self.controlLayout.setObjectName("controlLayout")
        self.trainBtn = QtWidgets.QPushButton(self.controlLayout)
        self.trainBtn.setGeometry(QtCore.QRect(0, 0, 90, 23))
        self.trainBtn.setObjectName("trainBtn")
        self.playBtn = QtWidgets.QPushButton(self.controlLayout)
        self.playBtn.setGeometry(QtCore.QRect(100, 0, 90, 23))
        self.playBtn.setObjectName("playBtn")
        self.showSelfCB = QtWidgets.QCheckBox(self.controlLayout)
        self.showSelfCB.setGeometry(QtCore.QRect(260, 0, 70, 17))
        self.showSelfCB.setObjectName("showSelfCB")
        self.showSelfCB.setCheckState(2)
        self.slider1 = QtWidgets.QSlider(self.centralwidget)
        self.slider1.setGeometry(QtCore.QRect(10, 230, 321, 22))
        self.slider1.setOrientation(QtCore.Qt.Horizontal)
        self.slider1.setObjectName("slider1")
        self.songLabel = QtWidgets.QLabel(self.centralwidget)
        self.songLabel.setGeometry(QtCore.QRect(10, 260, 231, 16))
        self.songLabel.setObjectName("songLabel")
        self.titleLabel = QtWidgets.QLabel(self.centralwidget)
        self.titleLabel.setGeometry(QtCore.QRect(20, 10, 231, 16))
        self.titleLabel.setObjectName("titleLabel")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 40, 311, 181))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 359, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.guiActivate()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", STRING_TITLE))
        self.trainBtn.setText(_translate("MainWindow", STRING_TRAIN))
        self.playBtn.setText(_translate("MainWindow", STRING_PLAY))
        self.showSelfCB.setText(_translate("MainWindow", STRING_SHOW_SELF))
        self.titleLabel.setText(_translate("MainWindow", STRING_LABEL))

    def guiActivate(self):
        #####Video part#####
        self.face_detection_widget = MoodDetectionWidget()
        self.record_video = RecordVideo()
        # Connect the image data signal and slot together
        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)
        # connect the run button to the start recording slot
        self.verticalLayout.addWidget(self.face_detection_widget)
        self.trainBtn.clicked.connect(self.train)
        self.playBtn.clicked.connect(self.play_pause)
        self.showSelfCB.stateChanged.connect(self.showSelf)
        #####Audio part#####
        self.audio_player = MoodPlayLists()
        #self.happy_song = QtCore.QUrl.fromLocalFile(HAPPY_SONG)
        #self.sad_song = QtCore.QUrl.fromLocalFile(SAD_SONG)
        mood_change_slot = self.mood_change_slot
        self.face_detection_widget.mood_change.connect(mood_change_slot)

    def showSelf(self):
        self.face_detection_widget.setVisible(self.showSelfCB.isChecked())

    def play(self):
        self.showSelf()
        self.record_video.start_recording()
        #content = QtMultimedia.QMediaContent(self.happy_song)
        #self.audio_player.setMedia(content)
        self.audio_player.play()
        songName = self.audio_player.currentMedia().canonicalUrl().fileName()
        self.songLabel.setText("Now Playing: %s"%(songName))
        self.playBtn.setText(STRING_PAUSE)

    def pause(self):
        self.record_video.stop_recording()
        self.audio_player.pause()
        self.playBtn.setText(STRING_PLAY)

    def play_pause(self):
        assert(self.playBtn.text() in [STRING_PLAY, STRING_PAUSE])
        if self.playBtn.text() == STRING_PLAY:
            self.play()
        else: #Pause
            self.pause()

    def zero_features(self):
        self.pause()
        if not self.showSelfCB.isChecked():
            self.showSelfCB.setChecked(True)
        self.record_video.start_recording()
        self.centralwidget.repaint()
        time.sleep(4)
        self.record_video.stop_recording()

    def train(self):
        # disable controls
        self.trainBtn.setEnabled(False)
        self.trainBtn.setText(STRING_TRAINING)
        self.playBtn.setEnabled(False)
        self.showSelfCB.setEnabled(False)
        self.controlLayout.repaint()
        # zero features
        self.zero_features()
        # reenable controls
        self.trainBtn.setText(STRING_TRAIN)
        self.trainBtn.setEnabled(True)
        self.playBtn.setEnabled(True)
        self.showSelfCB.setEnabled(True)

    def mood_change_slot(self, mood_change):
        # if mood_change == 1:
            # content = QtMultimedia.QMediaContent(self.happy_song)
        # else:
            # content = QtMultimedia.QMediaContent(self.sad_song)
        # self.audio_player.setMedia(content)
        self.audio_player.change_playlist(mood_change)
        self.audio_player.play()

class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)
    
    def stop_recording(self):
        self.timer.stop()
    
    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return
        read, image = self.camera.read()
        if read:
            self.image_data.emit(image)

class MoodDetectionWidget(QtWidgets.QWidget):
    mood_change = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.detector = dlib.get_frontal_face_detector()
        self.landmarks_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.mood_classifier = pickle.load(open(MOOD_PREDICTOR_FILENAME, 'rb'))
        self.image = QtGui.QImage()
        #Graphic properties
        self._red = (0, 0, 255)
        self._width = 2
        #features object
        self.features = FaceFeatures() #TODO adjust - need chen's part, init neutral features..
        #calculating delta between frames:
        self.prev_dists = []
        self.delta = 0
        self.is_first = True
        self.last_emotion = 0 #Start from neutral

    def is_frame_different(self, face_landmarks):
        """
        calculate frames delta - update prev dists if necessary
        output - (bool) delta > DELTA_THRESHOLD
        """
        #face_landmarks = np.array(face_landmarks[wanted_landmarks])
        dists = app_utils.extract_dist(face_landmarks)
        norm_factor = np.linalg.norm(face_landmarks[0]-face_landmarks[16])
        dists = dists / norm_factor
        delta = 0
        if not self.is_first:
            delta = np.linalg.norm(dists - self.prev_dists)
            if delta > DELTA_THRESHOLD:
                self.prev_dists = dists #Update when delta exceeding limit
                print("Debug - frames delta was %.2f"%(delta))
        else:
            self.is_first = False
            self.prev_dists = dists #First update
        return (delta > DELTA_THRESHOLD)
        
    def detect_faces(self, image: np.ndarray):
        gray = app_utils.preprocess_image(image)
        #gray = cv2.equalizeHist(gray)
        faces = self.detector(gray, 1)               # detect faces in the grayscale image (1?)
        return faces

    def image_data_slot(self, image_data):
        faces = self.detect_faces(image_data)
        image_data = cv2.resize(image_data, (350, 350))
        for (i, rect) in enumerate(faces):
            face_landmarks = self.landmarks_detector(app_utils.preprocess_image(image_data), rect)
            face_landmarks = app_utils.shape_to_np(face_landmarks)
            (x, y, w, h) = app_utils.rect_to_bb(rect) # convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x, y, w, h)]
            cv2.rectangle(image_data, (x, y), (x + w, y + h), self._red, self._width) #draw the face bounding box
            if self.is_frame_different(face_landmarks):
                #cv2.putText(image_data, "Delta from prev is {}".format(delta), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._red, self._width) #TODO delete (Debug)
                self.check_mood_change(face_landmarks)
        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())
        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage
        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()
        
    def check_mood_change(self, face_landmarks):
        """
        #TODO enable & check
        #1. Calculate Features and mood
        frame_features = self.features.extract_features(face_landmarks, False)
        mood = self.mood_classifier.predict(frame_features)
        #2. Check previous mood and update
        if (mood == self.last_emotion):
            return
        #3. (else) Send a signal if necessary 
        self.last_emotion = mood
        self.mood_change.emit(mood)
        """
        self.mood_change.emit(self.last_emotion)
        self.last_emotion = (self.last_emotion+1)%2

class FaceFeatures(object):
    def __init__(self):
        self.neutral_features = None

    def zero(self, face_landmarks):
        self.neutral_features = self.extract_features(face_landmarks, True)

    def extract_features(self, face_landmarks, is_first):
        """
        input - nparray of facial landmarks (point (x,y))
        output - nparray of features per image
        """
        #distance features
        dot_m = app_utils.dot_matrix(face_landmarks)
        dist_m = app_utils.dist_matrix(dot_m)
        dists = app_utils.dist_array(dist_m)
        norm_factor = np.linalg.norm(face_landmarks[0]-face_landmarks[3]) # dist(1,17)
        dists = dists / norm_factor
        #angles features
        angles = app_utils.angle_array(dot_m, dist_m)
        #flatten and concat
        features_vector = np.around(np.concatenate((dists, angles)),decimals = 2)
        #normalize
        if not is_first:
            features_vector = features_vector - self.neutral_features
        return features_vector

class MoodPlayLists(QtMultimedia.QMediaPlayer):
    #self.playlist.currentMediaChanged.connect(self.songChanged)
    def __init__(self, parent=None):
        super().__init__(parent)
        moods = glob.glob(PLAYLISTS_PATH +"//*")
        self.playlists = []
        for m in moods:
            songs = glob.glob(m + "//*")
            for s in songs:
                playlist = QMediaPlaylist(self)
                url = QtCore.QUrl.fromLocalFile(s)
                playlist.addMedia(QMediaContent(url))
            #playlist.setPlaybackMode(QMediaPlaylist.Loop)
            self.playlists.append(playlist)
        self.setPlaylist(self.playlists[0])

    def change_playlist(self, mood=0):
        self.setPlaylist(self.playlists[mood])
        #songName = self.currentMedia().canonicalUrl().fileName()
        #self.songLabel.setText("Now Playing: %s"%(songName))
    
    # def songChanged(self, media):
        # if not media.isNull():
            # url = media.canonicalUrl()
            # self.statusBar().showMessage(url.fileName())

class app_utils():
    def rect_to_bb(rect):
        """ take a bounding predicted by dlib and convert it
         to the format (x, y, w, h) as we would normally do
         with OpenCV
        """
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h) # return a tuple of (x, y, w, h)
        
    def shape_to_np(shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)     # initialize the list of (x, y)-coordinates
        # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords                               # return the list of (x, y)-coordinates
    
    def dot_matrix(point_arr):
        """
        input - nparray of (x, y) points
        output - an nxn matrix M where M[i, j] is the dot product of (xi, yi) and (xj, yj)
        """
        dot_m = np.ndarray(shape=(len(point_arr), len(point_arr)), dtype=int)
        for i in range(len(point_arr)):
            for j in range(i+1):
                dot_m[i, j] = np.dot(point_arr[i], point_arr[j])
                dot_m[j, i] = dot_m[i, j]
        return dot_m

    def dist_matrix(dot_m):
        """
        input - a dot matrix (output of dot_matrix method)
        output - an nxn matrix M where M[i, j] is the distance between (xi, yi) and (xj, yj)
        """
        dist_m = np.ndarray(shape=dot_m.shape, dtype=float)
        for i in range(dist_m.shape[0]):
            dist_m[i, i] = 0
            for j in range(i):
                dist_m[i, j] = np.sqrt(dot_m[i, i] - 2*dot_m[i, j] + dot_m[j, j])
                dist_m[j, i] = dist_m[i, j]
        return dist_m

    def dist_array(dist_m):
        """
        input - a distance matrix (output of dist_matrix method)
        output - an array of all of the distances, w/o duplicates
        """
        dists = []
        for i in range(dist_m.shape[0]):
            for j in range(i):
                dists.append(dist_m[i, j])
        return np.array(dists)

    def angle_array(dot_m, dist_m):
        """
        input - a dot matrix (output of dot_matrix method), a distance matrix (output of dist_matrix method)
        output - an array of all of the angles, w/o duplicates
        """
        angles = []
        for i in range(dot_m.shape[0]):
            for j in range(i):
                for k in range(j):
                    #TODO change solution to devision by 0
                    if not (dist_m[i, j] * dist_m[j, k] * dist_m[i, k]):
                        angles.append(-1)
                        angles.append(-1)
                        # angles.append(-1)
                    else:
                        angles.append(np.arccos(round(
                            (dot_m[i, k] - dot_m[i, j] - dot_m[j, k] + dot_m[j, j]) / (dist_m[i, j] * dist_m[j, k]),
                            15)))
                        angles.append(np.arccos(round(
                            (dot_m[i, j] - dot_m[i, k] - dot_m[k, j] + dot_m[k, k]) / (dist_m[i, k] * dist_m[k, j]),
                            15)))
                        # angles.append(np.pi - angles[-1] - angles[-2])
        return np.array(angles)
    
    def extract_dist(image):
        """
        input - nparray of facial landmarks (point (x,y))
        output - nparray of features per image
        """
        #distance features
        dot_m = app_utils.dot_matrix(image)
        dist_m = app_utils.dist_matrix(dot_m)
        dists = app_utils.dist_array(dist_m)
        return np.around(dists, decimals = 2)
        
    def preprocess_image(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (350, 350))
        return gray
        
def main():
    print("Debug - starting..")
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5()) # setup stylesheet
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    print("Debug - setup ui..")
    ui.setupUi(MainWindow)
    print("Debug - and..show!")
    MainWindow.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
