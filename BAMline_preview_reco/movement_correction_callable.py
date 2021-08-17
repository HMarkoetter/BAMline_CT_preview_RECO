# movement correction module
# 23.12.2020

#imports
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.uic import loadUiType
import tkinter.filedialog
from PIL import Image
import tomopy
import math
import time
import os
import scipy
from scipy import ndimage
from scipy import stats
from scipy import signal
import qimage2ndarray
import pyqtgraph as pg

pg.setConfigOption('background', 'w')  # Plothintergrund weiß (2D)
pg.setConfigOption('foreground', 'k')  # Plotvordergrund schwarz (2D)
pg.setConfigOptions(antialias=True)  # Enable antialiasing for prettier plots


Ui_CorrectionWindow, QCorrectionWindow = loadUiType('movement_correction_callable.ui')  # GUI vom Hauptfenster


class Movement_corrector(Ui_CorrectionWindow, QCorrectionWindow):


    def __init__(self, sinogram, path_out, namepart, number_of_sequences, sequence_size, cor, max_theta, theta_first_list):
        super(Movement_corrector, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Movement Correction Callable')
        self.movement_correction_running = 1
        #QtWidgets.QApplication.processEvents()
        sinogram = (sinogram * 0.001) - 9.68
        self.multi_sino = numpy.exp(numpy.negative(sinogram))

        self.sino = numpy.zeros(sinogram.shape, dtype=float)      #, dtype=float) #, dtype=numpy.int16
        self.full_size = self.multi_sino.shape[0]
        print('full_size', self.full_size)

        self.path_out = path_out
        self.namepart = namepart
        self.number_of_sequences = number_of_sequences
        self.sequence_size = sequence_size
        self.COR = cor
        self.max_theta = max_theta
        self.theta_first_list = theta_first_list

        self.pushAnalyze.clicked.connect(self.analyze)
        self.pushCorrect.clicked.connect(self.correct)
        self.pushReconstruct.clicked.connect(self.reconstruct)
        self.pushApplyVolume.clicked.connect(self.apply_to_volume)
        self.pushDoNotApply.clicked.connect(self.do_not_apply)
        #self.brightness.valueChanged.connect(self.show_reco)

        self.number_of_projections = self.sequence_size * self.number_of_sequences
        print(self.number_of_projections, 'projections total')

        self.path_out_corr = self.path_out + '/Movement_Correction'

        if os.path.isdir(self.path_out_corr) is False:
            os.mkdir(self.path_out_corr)
            print('creating path_out_corr', self.path_out_corr)

        if os.path.isdir(self.path_out_corr + '/Sinogram_comparison' ) is False:
            os.mkdir(self.path_out_corr + '/Sinogram_comparison' )
            print('creating path_out_corr', '/Sinogram_comparison' , self.path_out_corr , '/Sinogram_comparison' )

        # prepare sinogram, reform to one long instead of several short sinograms
        n = 1
        while n < self.number_of_projections + 1:
            print(n, int(self.number_of_sequences * ((n-1) % self.sequence_size)  +  (self.theta_first_list[math.floor((n-1)/self.sequence_size)] - min(self.theta_first_list)) * self.number_of_sequences ))
            self.progressBar.setValue((n + 1) * 100 / self.number_of_projections)
            QtWidgets.QApplication.processEvents()

            self.sino[:, int(self.number_of_sequences * ((n-1) % self.sequence_size)  +  (self.theta_first_list[math.floor((n-1)/self.sequence_size)] - min(self.theta_first_list)) * self.number_of_sequences)] = self.multi_sino[:,n-1]
            n = n + 1


        img = Image.fromarray(self.sino)
        self.filename_out = self.path_out_corr + '/Sinogram_comparison' + self.namepart + 'sinogram_original' + '.tif'
        img.save(self.filename_out)

        myarray = self.sino * 256 / 16000 # * contrast - (contrast - 128)  # 2048 - 1920
        myarray = myarray.repeat(2, axis=0).repeat(2, axis=1)
        yourQImage = qimage2ndarray.array2qimage(myarray)
        self.original_sino.setPixmap(QPixmap(yourQImage))

        print('movement corrector init done')


    def analyze(self):

        self.pushAnalyze.setEnabled(False)
        self.scan_range.setEnabled(False)

        scan_range = self.scan_range.value()

        self.step_size_ = self.step_size.currentText()
        if self.step_size_ == '1':
            stp = 1
            pts = 1
            print('Step Size ',stp)

        if self.step_size_ == '0.5':
            stp = 0.5
            pts = 2
            print('Step Size ',stp)

        if self.step_size_ == '0.25':
            stp = 0.25
            pts = 4
            print('Step Size ',stp)

        if self.step_size_ == '0.1':
            stp = 0.1
            pts = 10
            print('Step Size ',stp)


        results_ = numpy.zeros(2 * scan_range * pts + 1)
        best = numpy.zeros(self.number_of_projections)
        self.summe = numpy.zeros(self.number_of_projections)
        print('results_', results_.shape, 'best', best.shape, 'summe', self.summe.shape)


        if self.max_theta < 270:
            x = numpy.arange(self.number_of_projections) * 180 / self.number_of_projections
        else:
            x = numpy.arange(self.number_of_projections) * 360 / self.number_of_projections


        # ANALYZING DATA #
        n = 1
        while n < self.number_of_projections:
            self.progressBar.setValue((n+1) * 100 / self.number_of_projections)
            QtWidgets.QApplication.processEvents()

            image1 = self.sino[:, n - 1]
            image2 = self.sino[:, n]
            image2 = numpy.nan_to_num(image2, copy=True, nan=1.0, posinf=1.0, neginf=1.0)

            k = 0
            while k < 2 * scan_range * pts + 0.01:
                image2_shifted = ndimage.shift(image2, k * stp - scan_range,  mode='nearest', prefilter=True)
                dummy = scipy.stats.pearsonr(image2_shifted, image1)
                results_[k] = dummy[0]
                k = k + 1

            best[n] = (numpy.argmax(results_) * stp - scan_range)
            self.summe[n] = self.summe[n-1] + best[n]
            print('analyze n',n,'best',(numpy.argmax(results_) * stp - scan_range), '      summe', self.summe[n])

            self.Graph.setLabel('left', 'Accumulated Shift (pixel)')
            self.Graph.setLabel('bottom', 'Deg (°)')
            self.Graph.plot(x[:n], self.summe[:n], pen='b', clear=True)

            n = n + 1

        print('summe',self.summe.shape)#, self.summe)


        self.pushCorrect.setEnabled(True)
        self.filter_list.setEnabled(True)
        self.savgol_window.setEnabled(True)
        self.savgol_poly.setEnabled(True)
        self.pushAnalyze.setEnabled(True)
        self.scan_range.setEnabled(True)



    def correct(self):
        self.pushCorrect.setEnabled(False)
        self.filter_list.setEnabled(False)
        self.savgol_window.setEnabled(False)
        self.savgol_poly.setEnabled(False)
        self.pushAnalyze.setEnabled(False)
        self.scan_range.setEnabled(False)


        QtWidgets.QApplication.processEvents()
        print('def_correct')
        # extend to avoid edge handling
        extended = numpy.zeros(3 * self.number_of_projections, dtype=float)
        print('extended',extended.shape)#, extended)
        print(extended[0 : self.number_of_projections].shape)
        extended[0 : self.number_of_projections] = self.summe
        print('extended',extended.shape)#, extended)
        extended[self.number_of_projections : 2 * self.number_of_projections] = self.summe +   self.summe[-1]
        print('extended',extended.shape)#, extended)
        extended[2 * self.number_of_projections : 3 * self.number_of_projections] = self.summe + 2*self.summe[-1]
        print('extended',extended.shape)#, extended)


        if (self.savgol_window.value() % 2) != 1:
            even_value = self.savgol_window.value()
            self.savgol_window.setValue(even_value + 1)

        self.filter = self.filter_list.currentText()
        if self.filter == 'Savitzky_Golay':
            filtered_trace = signal.savgol_filter(extended, self.savgol_window.value(), self.savgol_poly.value())
            print('Applying Savitzky-Golay')

        if self.filter == 'Low_Pass':
            kern = signal.firwin(31, cutoff = 1 / (2 * self.number_of_sequences) , window="hamming")
            filtered_trace = Smooth = signal.lfilter(kern, 1, extended)
            print('Applying Low-Pass')

        # cut from extended
        filtered_trace_cut = filtered_trace[self.number_of_projections : 2 * self.number_of_projections] - self.summe[-1]
        print('summe', self.summe.shape)#, self.summe)
        print('filtered_trace_cut', filtered_trace_cut.shape)#, filtered_trace_cut)
        self.correction_list = self.summe - filtered_trace_cut
        print('correction_list', self.correction_list.shape)#, self.correction_list)

        if self.max_theta < 270:
            x = numpy.arange(self.number_of_projections) * 180 / self.number_of_projections
        else:
            x = numpy.arange(self.number_of_projections) * 360 / self.number_of_projections


        pen = pg.mkPen(color=(255, 0, 0), width=5)#, style=QtCore.Qt.DashLine)
        self.Graph.plot(x, self.summe, name = "original", pen='b', clear=True)
        self.Graph.plot(x, filtered_trace_cut, name = "corrected", pen=pen)
        self.Graph.plot(x, self.correction_list, name = "adjustment", pen='g')

        self.corrected_sinogram = numpy.zeros((self.full_size, self.number_of_projections), dtype=float)
        n = 0
        while n < self.number_of_projections:
            self.progressBar.setValue((n+1) * 100 / self.number_of_projections)
            QtWidgets.QApplication.processEvents()

            #print('correcting n', n, self.sino.shape)
            self.corrected_sinogram[:, n] = ndimage.shift(self.sino[:, n], self.correction_list[n],  mode='nearest', prefilter=True)
            n = n + 1
        print('corrected_sinogram', self.corrected_sinogram.shape)

        img = Image.fromarray(self.corrected_sinogram)
        self.filename_out2 = self.path_out_corr + '/Sinogram_comparison'+ self.namepart + 'sinogram_corrected' + '.tif'
        img.save(self.filename_out2)

        print('saved corrected sinogram')
        print('prepare display')

        myarray = self.corrected_sinogram * 256 / 16000  # * contrast - (contrast - 128)  # 2048 - 1920
        myarray = myarray.repeat(2,axis=0).repeat(2,axis=1)
        yourQImage = qimage2ndarray.array2qimage(myarray)
        self.corrected_sino.setPixmap(QPixmap(yourQImage))

        print('done')
        self.pushReconstruct.setEnabled(True)
        self.pushCorrect.setEnabled(True)
        self.filter_list.setEnabled(True)
        self.savgol_window.setEnabled(True)
        self.savgol_poly.setEnabled(True)
        self.pushAnalyze.setEnabled(True)
        self.pushApplyVolume.setEnabled(True)
        self.scan_range.setEnabled(True)


    def reconstruct(self):
        self.pushReconstruct.setEnabled(False)
        self.pushAnalyze.setEnabled(False)
        self.scan_range.setEnabled(False)
        self.pushCorrect.setEnabled(False)
        self.filter_list.setEnabled(False)
        self.savgol_window.setEnabled(False)
        self.savgol_poly.setEnabled(False)

        QtWidgets.QApplication.processEvents()
        print('def reconstruct')

        if self.max_theta < 270:
            new_list = numpy.arange(self.number_of_projections) * math.pi / self.number_of_projections
            print('180 Deg Scan detected')
        else:
            new_list = numpy.arange(self.number_of_projections) * 2 * math.pi / self.number_of_projections
            print('360 Deg Scan detected')


        center_list = [self.COR + round(0.5 * self.full_size)] * (self.number_of_projections)

        sinos = numpy.zeros((2, self.full_size, self.number_of_projections), dtype=float)
        sinos[0,:,:] = self.sino
        sinos[1,:,:] = self.corrected_sinogram
        transposed_sinos = numpy.transpose(sinos, axes = [2,0,1])
        print('transposed_sinos_shape', transposed_sinos.shape)

        extended_sinos = tomopy.misc.morph.pad(transposed_sinos, axis=2, npad=round(0.5 * self.full_size), mode='edge')
        extended_sinos = tomopy.minus_log(extended_sinos)
        extended_sinos = (extended_sinos + 9.68) * 1000

        slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm='gridrec', filter_name='shepp')
        slices = slices[:,round(0.5 * self.full_size) : -round(0.5 * self.full_size) , round(0.5 * self.full_size) : -round(0.5 * self.full_size)]
        slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)

        slices = (slices + 100) * 320
        slices = numpy.clip(slices, 1, 65534)
        slices = slices.astype(numpy.uint16)

        self.corrected_reconstruction = slices[1, :, :]
        self.original_reconstruction = slices[0, :, :]
        print('reconstructions done')

        if os.path.isdir(self.path_out_corr + '/Reconstruction_comparison' ) is False:
            os.mkdir(self.path_out_corr + '/Reconstruction_comparison' )
            print('creating path_out_corr', '/Reconstruction_comparison' , self.path_out_corr , '/Reconstruction_comparison' )

        img = Image.fromarray(self.corrected_reconstruction)
        self.filename_out3 = self.path_out_corr + '/Reconstruction_comparison' + self.namepart + 'reconstruction_corrected' + '.tif'
        img.save(self.filename_out3)
        img = Image.fromarray(self.original_reconstruction)
        self.filename_out4 = self.path_out_corr + '/Reconstruction_comparison' + self.namepart + 'reconstruction_original' + '.tif'
        img.save(self.filename_out4)
        print('reconstructions saved')


        myarray = (self.corrected_reconstruction - 31000) * 0.005 * self.brightness.value()  # * contrast - (contrast - 128)  # 2048 - 1920
        #myarray = (ima16 - 31000) * 0.005 * self.BrightnessSlider.value()
        myarray = myarray.repeat(2, axis=0).repeat(2, axis=1)
        yourQImage = qimage2ndarray.array2qimage(myarray)
        self.corrected_reco.setPixmap(QPixmap(yourQImage))

        myarray = (self.original_reconstruction - 31000) * 0.005 * self.brightness.value()
        #myarray = self.original_reconstruction * self.brightness.value()  # * contrast - (contrast - 128)  # 2048 - 1920
        myarray = myarray.repeat(2, axis=0).repeat(2, axis=1)
        yourQImage = qimage2ndarray.array2qimage(myarray)
        self.original_reco.setPixmap(QPixmap(yourQImage))


        self.tabWidget.setCurrentIndex(4)
        self.pushReconstruct.setEnabled(True)
        self.pushAnalyze.setEnabled(True)
        self.scan_range.setEnabled(True)
        self.pushCorrect.setEnabled(True)
        self.filter_list.setEnabled(True)
        self.savgol_window.setEnabled(True)
        self.savgol_poly.setEnabled(True)
        self.pushApplyVolume.setEnabled(True)
        print('reconstructions displayed')
        #self.brightness.setEnabled(True)
        #self.show_reco()



    #def show_reco(self):




    def apply_to_volume(self):
        print('apply_to_volume')
        self.movement_corrector_results = self.correction_list
        print('self.movement_corrector_results = self.correction_list')

        self.checkBox_save_shifted_projections = self.checkBox_save_shifted_projections.isChecked()

        self.movement_correction_running = 0
        print('self.movement_correction_running', self.movement_correction_running)
        self.close()
        print('self.close()')


    def do_not_apply(self):
        print('do_not_apply')
        self.movement_corrector_results = numpy.zeros(self.number_of_sequences * self.sequence_size)
        print('self.movement_corrector_results = numpy.zeros(self.number_of_sequences * self.sequence_size)')
        self.movement_correction_running = 0
        print('self.movement_correction_running', self.movement_correction_running)
        self.close()
        print('self.close()')



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    main = Movement_corrector()
    main.show()
    sys.exit(app.exec_())