# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'WelcomeWindow.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

MAX_NAME_LEN = 15

class Ui_WelcomeWindow(object):
    def setupUi(self, WelcomeWindow):
        WelcomeWindow.setObjectName("WelcomeWindow")
        WelcomeWindow.resize(296, 200)
        self.centralwidget = QtWidgets.QWidget(WelcomeWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.FaceTheMusicLabel = QtWidgets.QLabel(self.centralwidget)
        self.FaceTheMusicLabel.setGeometry(QtCore.QRect(40, 0, 261, 61))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(26)
        self.FaceTheMusicLabel.setFont(font)
        self.FaceTheMusicLabel.setTextFormat(QtCore.Qt.AutoText)
        self.FaceTheMusicLabel.setObjectName("FaceTheMusicLabel")
        self.HaveWeMetLabel = QtWidgets.QLabel(self.centralwidget)
        self.HaveWeMetLabel.setGeometry(QtCore.QRect(10, 70, 111, 16))
        self.HaveWeMetLabel.setObjectName("HaveWeMetLabel")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 100, 131, 16))
        self.label.setObjectName("label")
        self.NameLineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.NameLineEdit.setGeometry(QtCore.QRect(170, 100, 113, 20))
        self.NameLineEdit.setObjectName("NameLineEdit")
        self.GoButton = QtWidgets.QPushButton(self.centralwidget)
        self.GoButton.setGeometry(QtCore.QRect(100, 140, 75, 23))
        self.GoButton.setObjectName("GoButton")
        WelcomeWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(WelcomeWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 296, 21))
        self.menubar.setObjectName("menubar")
        WelcomeWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(WelcomeWindow)
        self.statusbar.setObjectName("statusbar")
        WelcomeWindow.setStatusBar(self.statusbar)
        self.ErrorLabel = QtWidgets.QLabel(self.centralwidget)
        self.ErrorLabel.setGeometry(QtCore.QRect(10, 160, 150, 16))
        self.ErrorLabel.setObjectName("ErrorLabel")

        self.retranslateUi(WelcomeWindow)
        QtCore.QMetaObject.connectSlotsByName(WelcomeWindow)

    def retranslateUi(self, WelcomeWindow):
        _translate = QtCore.QCoreApplication.translate
        WelcomeWindow.setWindowTitle(_translate("WelcomeWindow", "Welcome to Face The Music"))
        self.FaceTheMusicLabel.setText(_translate("WelcomeWindow", "Face The Music "))
        self.HaveWeMetLabel.setText(_translate("WelcomeWindow", "Have we met before?"))
        self.label.setText(_translate("WelcomeWindow", "Please enter your name:"))
        self.GoButton.setText(_translate("WelcomeWindow", "Let\'s go!"))
        
    def guiActivate(self):
        self.GoButton.clicked.connect(self.letsgo)
        self.ErrorLabel.hide()
        self.ErrorLabel.setText("Invalid Name. Please try again.")
        self.ErrorLabel.setStyleSheet('color: red')
        
    def letsgo(self):
        name = self.NameLineEdit.text()
        if not name.isalpha() or len(name)>MAX_NAME_LEN:
            print("invalid name: "+ name)
            self.ErrorLabel.show()
        else:
            self.ErrorLabel.hide()
            print("Name is " + name)
            QtCore.QCoreApplication.instance().quit()






if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    WelcomeWindow = QtWidgets.QMainWindow()
    ui = Ui_WelcomeWindow()
    ui.setupUi(WelcomeWindow)
    ui.guiActivate()
    WelcomeWindow.show()
    sys.exit(app.exec_())

