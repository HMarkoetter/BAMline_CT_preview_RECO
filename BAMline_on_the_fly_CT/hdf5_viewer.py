

import h5py
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUiType


dialog = QtGui.QFileDialog(self, 'Audio Files', directory, filter)
dialog.setFileMode(QtGui.QFileDialog.DirectoryOnly)
dialog.setSidebarUrls([QtCore.QUrl.fromLocalFile(place)])
if dialog.exec_() == QtGui.QDialog.Accepted:
    self._audio_file = dialog.selectedFiles()[0]



#filename = "file.hdf5"
path_klick_FF = QtWidgets.QFileDialog.getOpenFileName( 'Select first FF-file, please.', "C:\Fly_and_Helix_Test\Flying-CT_Test\MEA_Flying-CT_2x2_crop_normalized")

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    #data = list(f[a_group_key])