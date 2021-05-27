# reConstruction Site for live-reconstruction

import numpy
from scipy import ndimage
from PIL import Image
import math
import tomopy
import time
from time import localtime, strftime
import tkinter as Tkinter
import tkinter.filedialog
import os
import qimage2ndarray
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QIcon, QPixmap, QCloseEvent
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import csv
from movement_correction_callable import Movement_corrector


Ui_CT_previewWindow, QCOR_previewWindow = loadUiType('CT-preview.ui')  # GUI vom Hauptfenster


class CT_preview(Ui_CT_previewWindow, QCOR_previewWindow):

    def __init__(self, COR, rotate, pixel_size, path_klick, path_out, block_size, dark_field_value, no_of_cores, checkBox_save_normalized, checkBox_classic_order, transpose, find_pixel_size_vertical):#, algorithm, filter):
        super(CT_preview, self).__init__()
        self.setupUi(self)

        self.setWindowTitle('Interlaced-CT Preview Reconstruction')

        self.COR = COR
        self.rotate = rotate
        self.COR_change.setValue(self.COR)
        self.pixel_size = pixel_size

        self.path_klick = path_klick
        self.path_out = path_out
        self.block_size = block_size
        self.dark_field_value = dark_field_value
        self.no_of_cores = no_of_cores

        self.checkBox_save_normalized = checkBox_save_normalized
        self.checkBox_classic_order = checkBox_classic_order
        self.transpose = transpose
        self.find_pixel_size_vertical = find_pixel_size_vertical



        self.Start.clicked.connect(self.mains)
        #self.mains()

        print(COR)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'COR ' + str(COR))
        print(rotate)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'rotate ' + str(rotate))
        print(pixel_size)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'pixel_size ' + str(pixel_size))
        print(dark_field_value)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'dark_field_value ' + str(dark_field_value))
        print(no_of_cores)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'no_of_cores ' + str(no_of_cores))
        print(checkBox_classic_order)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'checkBox_classic_order ' + str(checkBox_classic_order))




    def mains(self):

        self.Preview_slice.setEnabled(True)
        self.preview_frequency.setEnabled(True)

        slice = self.Preview_slice.value()              # Preview Slice                                                                     # Whats wrong here? #
        extend_FOV_base = 0.05

        if self.checkBox_save_normalized == True:
            save_normalized = 1                         # save normalized projections?
        else:
            save_normalized = 0

        if self.checkBox_classic_order == True:
            save_normalized_classic_order = 1           # sorts the normalized projections to steadily rising angles !!! SET THIS 0 FOR CLASSIC-CT !!!
        else:
            save_normalized_classic_order = 0

        preview_frequency = 10                          # preview reconstruction on idle and every 5th, 10th, 20th or 50th projection
        volume_begin = 0                                # reconstruct these slices in the end


        # DEFINING PATHS # ==================================================================================================== # DEFINING PATHS #

        root = Tkinter.Tk()

        path_out = self.path_out

        path_klick = self.path_klick
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_klick)
        htap = self.path_klick[::-1]
        path_in = self.path_klick[0: len(htap) - htap.find('/') - 1: 1]
        namepart = self.path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 5: 1]
        counter = self.path_klick[len(htap) - htap.find('.') - 5: len(htap) - htap.find('.') - 1:1]
        filetype = self.path_klick[len(htap) - htap.find('.') - 1: len(htap):1]


        print(path_out)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_out)

        path_lists = path_in
        print(path_lists)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_lists)
        root.withdraw()

        path_out_reconstructed = path_out + '/Prev'
        path_out_reconstructed_full = path_out + '/Vol'
        path_out_normalized = path_out + '/Norm'
        path_out_changes = path_out + '/Changes'

        print(path_out_reconstructed)
        print(path_out_reconstructed_full)
        print(path_out_normalized)
        print(path_out_changes)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_out_reconstructed)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_out_reconstructed_full)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_out_normalized)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_out_changes)

        if os.path.isdir(path_out_reconstructed) is False:
            os.mkdir(path_out_reconstructed)
        if os.path.isdir(path_out_reconstructed_full) is False:
            os.mkdir(path_out_reconstructed_full)
        if save_normalized == 1:
            if os.path.isdir(path_out_normalized) is False:
                os.mkdir(path_out_normalized)
        if os.path.isdir(path_out_changes) is False:
            os.mkdir(path_out_changes)
        if os.path.isdir(path_out_changes + '/Zero_Deg') is False:
            os.mkdir(path_out_changes + '/Zero_Deg')
        if os.path.isdir(path_out_changes + '/Zero_Deg_divided') is False:
            os.mkdir(path_out_changes + '/Zero_Deg_divided')
        if os.path.isdir(path_out_changes + '/Paulchen_Normalized') is False:
            os.mkdir(path_out_changes + '/Paulchen_Normalized')
        if os.path.isdir(path_out_changes + '/Paulchen_Divided') is False:
            os.mkdir(path_out_changes + '/Paulchen_Divided')


        first = 1   # HIER ERSTE DATEI DES SCANS EINTRAGEN ; BEI MEHREREN SCANS WICHTIG

        self.file_name_protocol = path_out + '/' + 'reconstruction_protocol.txt'

        # READ THETA AND X_OFFSET-LIST # ====================================================================================== # READ THETA AND X_OFFSET-LIST #

        file_name_theta = path_lists + '/theta_list.txt'
        theta_list = numpy.genfromtxt(file_name_theta)
        file_name_X_offset = path_lists + '/X_offset_list.txt'
        x_offset_list = numpy.genfromtxt(file_name_X_offset)
        file_name_theta_first_list = path_lists + '/theta_first_list.txt'
        theta_first_list = numpy.genfromtxt(file_name_theta_first_list)
        file_name_parameter = path_lists + '/parameter.csv'
        print(file_name_parameter)

        f = open(file_name_parameter, 'r')              # Reading scan-scheme parameters
        for line in f:
            line = line.strip()
            columns = line.split()
            print(columns[0])

            if str(columns[0]) == 'box_lateral_shift':
                box_lateral_shift = int(columns[1])
                print(columns[1])

            if str(columns[0]) == 'number_of_sequences':
                number_of_sequences = int(columns[1])
                print(columns[1])

            if str(columns[0]) == 'sequence_size':
                sequence_size = int(columns[1])
                print(columns[1])

            if str(columns[0]) == 'FF_sequence_size':
                FF_sequence_size = int(columns[1])
                print(columns[1])

            if str(columns[0]) == 'zero_deg_proj':
                zero_deg_proj = int(columns[1])
                print(columns[1])


        f.close()


        number_of_projections = sequence_size * number_of_sequences
        last = first + number_of_projections - 1

        self.preview_frequency.setValue(round(sequence_size/2))

        # OPEN AND NORMALIZE 0, 90 AND 180° # ============================================================================= # OPEN AND NORMALIZE 0, 90 AND 180° #

        print(self.algorithm_list.currentText(), 'algorithm chosen')
        print(self.filter_list.currentText(), 'filter chosen')
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + str(self.algorithm_list.currentText()) + 'algorithm chosen')
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + str(self.filter_list.currentText()) + 'filter chosen')
        filename1 = path_in + namepart + str(first).zfill(4) + filetype
        filename2 = path_in + namepart + str(first + 1).zfill(4) + filetype
        filename3 = path_in + namepart + str(first + 2).zfill(4) + filetype
        filename4 = path_in + namepart + str(first + 3).zfill(4) + filetype
        filename5 = path_in + namepart + str(first + 4).zfill(4) + filetype

        while os.path.exists(filename5) != True:
            time.sleep(3)
            print('waiting for next file:', filename4)
            self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'waiting for next file:', filename4)
        print('Reading data of 0, 90 and 180°')
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Reading data of 0, 90 and 180°')

        im_000deg = Image.open(filename1)
        im_090deg = Image.open(filename2)
        im_180deg = Image.open(filename3)
        FF = Image.open(filename4)

        if self.transpose == True:
            im_000deg = im_000deg.transpose(Image.TRANSPOSE)
            im_090deg = im_090deg.transpose(Image.TRANSPOSE)
            im_180deg = im_180deg.transpose(Image.TRANSPOSE)
            FF = FF.transpose(Image.TRANSPOSE)

        im = im_000deg
        DF = numpy.ones((im_000deg.size[1], im_000deg.size[0]), numpy.float32)
        DF = DF * self.dark_field_value
        DF = numpy.single(DF)
        im_000deg = numpy.single(numpy.array(im_000deg))
        im_090deg = numpy.single(numpy.array(im_090deg))
        im_180deg = numpy.single(numpy.array(im_180deg))
        proj_000_sub = numpy.subtract(im_000deg, DF)
        proj_090_sub = numpy.subtract(im_090deg, DF)
        proj_180_sub = numpy.subtract(im_180deg, DF)
        FF_sub = numpy.subtract(numpy.array(FF), numpy.array(DF))
        im_000_normalized = numpy.divide(proj_000_sub, FF_sub)
        im_090_normalized = numpy.divide(proj_090_sub, FF_sub)
        im_180_normalized = numpy.divide(proj_180_sub, FF_sub)

        filename_b_000 = path_out_changes + '/Paulchen_Normalized' + namepart + 'beginning_000_deg' + filetype
        print('Beginning Projection at 0°:', filename_b_000)
        self.logbook.append(
            strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Beginning Projection at 0°:' + filename_b_000)
        img = Image.fromarray(im_000_normalized)
        img.save(filename_b_000)

        filename_b_090 = path_out_changes + '/Paulchen_Normalized' + namepart + 'beginning_090_deg' + filetype
        print('Beginning Projection at 90°:', filename_b_090)
        self.logbook.append(
            strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Beginning Projection at 90°:' + filename_b_090)
        img = Image.fromarray(im_090_normalized)
        img.save(filename_b_090)

        filename_b_180 = path_out_changes + '/Paulchen_Normalized' + namepart + 'beginning_180_deg' + filetype
        print('Beginning Projection at 180°:', filename_b_180)
        self.logbook.append(
            strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Beginning Projection at 180°:' + filename_b_180)
        img = Image.fromarray(im_180_normalized)
        img.save(filename_b_180)

        print('image size', im_000deg.shape[0], 'x', im_000deg.shape[1], ' Total number of projections',
              last - first + 1)
        self.logbook.append(
            strftime("%Y_%m_%d %H:%M:%S ", localtime()) + ' Image size ' + str(im_000deg.shape[0]) + ' x ' + str(
                im_000deg.shape[1]) + ' Total number of projections' + str(last - first + 1))

        self.Preview_slice.setRange(2, im.size[1] - 2)
        self.Preview_slice.setValue(round(im.size[1] / 2))

        if slice > im.size[1] - 1:
            self.Preview_slice.setValue(round(im.size[1] / 2))
            print('Slice out of bound! Slice set to', round(im.size[1] / 2))
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + ' Slice out of bound! Slice set to ' + str(
                    round(im.size[1] / 2)))
            time.sleep(0.5)


        extend_FOV = (abs(self.COR - im.size[0]/2))/ (1 * im.size[0]) + extend_FOV_base    # extend field of view (FOV), 0.0 no extension, 0.5 half extension to both sides (for half sided 360 degree scan!!!)
        print('extend_FOV ', extend_FOV)



        # CALCULATE CENTER OF ROTATION # ================================================================================== # CALCULATE CENTER OF ROTATION #
        # do I need this at all?

        print('used cor:', self.COR)

        FF0 = numpy.ones((FF_sequence_size, im.size[1], im.size[0]), numpy.float32)
        print('stack size: number of images', last - first + 1, '; Y =', im.size[1], '; X =', im.size[0])
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + ' stack size: number of images ' + str(
            last - first + 1) + '; Y =' + str(im.size[1]) + '; X =' + str(im.size[0]))

        # FOLLOW AND READ DATA AND RECONSTRUCT # ===========================================================================# FOLLOW AND READ DATA AND RECONSTRUCT #

        i = first  # Projection number  (EXcluding FFs, zero, Paulchen, etc)                                         # Projection number counter
        n = 4  # Image file number       (INcluding FFs, zero, Paulchen, etc)                                        # file number counter
        while i < last + 1:

            self.lcdNumber_Total.display(i)
            self.lcdNumber_Image.display(i % sequence_size)
            self.lcdNumber_Sequence.display(math.ceil(i / sequence_size))
            self.progressBar_Sequence.setValue((i % sequence_size) * 100 / sequence_size)
            self.progressBar_Total.setValue(i * 100 / (sequence_size * number_of_sequences))

            QtCore.QCoreApplication.processEvents()
            time.sleep(0.2)

            # FF u. zero deg
            if (i % sequence_size) == 1:

                j = 1  # FF counter
                while (j < FF_sequence_size + 1):
                    filename_FF = path_in + namepart + str(n).zfill(4) + filetype
                    filename_FF_ = path_in + namepart + str(n + 1).zfill(4) + filetype

                    while os.path.exists(filename_FF_) != True:
                        time.sleep(2)
                        print('Waiting for next Flat Field:', filename_FF)
                        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + ' Waiting for next Flat Field: ' + filename_FF)
                        QtCore.QCoreApplication.processEvents()
                        time.sleep(0.02)
                        if self.Abort_and_reconstruct_now.isChecked() == True:
                            break
                    if self.Abort_and_reconstruct_now.isChecked() == True:
                        break
                    # time.sleep(0.2)
                    print('Loading FF ', filename_FF)
                    self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Loading FF ' + filename_FF)
                    im_FF = Image.open(filename_FF)

                    if self.transpose == True:
                        im_FF = im_FF.transpose(Image.TRANSPOSE)

                    FF0[j - 1, :, :] = numpy.array(im_FF)
                    n = n + 1
                    j = j + 1
                    if (j == FF_sequence_size):
                        FF_avg = numpy.median(FF0, axis=0)
                        FF_avg = numpy.single(FF_avg)


                # Waiting for next file
                filename_zero_load_waitfile = path_in + namepart + str(n + 1).zfill(4) + filetype
                while os.path.exists(filename_zero_load_waitfile) != True:
                    time.sleep(2)
                    print('Waiting for next Flat Field:', filename_FF)
                    self.logbook.append(
                        strftime("%Y_%m_%d %H:%M:%S ", localtime()) + ' Waiting for next Flat Field: ' + filename_FF)
                    QtCore.QCoreApplication.processEvents()
                    time.sleep(0.02)
                    if self.Abort_and_reconstruct_now.isChecked() == True:
                        break


                # ZERO DEG CHECK OR SKIP
                if zero_deg_proj == 2:
                    filename_zero_load = path_in + namepart + str(n).zfill(4) + filetype
                    filename_zero = path_out_changes + '/Zero_Deg' + namepart + str(i - first).zfill(4) + filetype
                    filename_zero_divided = path_out_changes + '/Zero_Deg_divided' + namepart + str(i - first).zfill(4) + filetype

                    print('Loading Zero Degree Projection ', filename_zero_load)
                    self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Loading Zero Degree Projection ' + filename_zero_load)
                    im_zero = Image.open(filename_zero_load)

                    if self.transpose == True:
                        im_zero = im_zero.transpose(Image.TRANSPOSE)

                    im_zero = numpy.single(numpy.array(im_zero))
                    im_zero_sub = numpy.subtract(im_zero, DF)
                    FF_sub = numpy.subtract(numpy.array(FF_avg), numpy.array(DF))
                    im_zero_normalized = numpy.divide(im_zero_sub, FF_sub)

                    im_zero_normalized = numpy.nan_to_num(im_zero_normalized, copy=True, nan=1.0, posinf=1.0, neginf=1.0)

                    #im_zero_normalized = ndimage.shift(numpy.single(numpy.array(im_zero_normalized)), [0, (x_offset_list[i - first] / self.pixel_size)],
                    #                    order=3, mode='nearest', prefilter=True)
                    if self.find_pixel_size_vertical == True:
                        im_zero_normalized = ndimage.shift(numpy.single(numpy.array(im_zero_normalized)),
                                            [-(x_offset_list[i - first] / self.pixel_size), 0],
                                            order=3, mode='nearest', prefilter=True)
                    else:
                        im_zero_normalized = ndimage.shift(numpy.single(numpy.array(im_zero_normalized)),
                                            [0, (x_offset_list[i - first] / self.pixel_size)],
                                            order=3, mode='nearest', prefilter=True)

                    im_zero_divided = numpy.divide(im_zero_normalized, im_000_normalized)

                    print('Writing Zero Degree Projection ', filename_zero)
                    self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ",
                                                 localtime()) + 'Writing Zero Degree Projection ' + filename_zero)
                    im_zero_normalized = Image.fromarray(im_zero_normalized)
                    im_zero_normalized.save(filename_zero)
                    
                    print('Writing divided Zero Degree Projection ', filename_zero_divided)
                    self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ",
                                                 localtime()) + 'Writing divided Zero Degree Projection ' + filename_zero_divided)
                    im_zero_divided = Image.fromarray(im_zero_divided)
                    im_zero_divided.save(filename_zero_divided)

                    n = n + 1


                    drift_correction = 0 # in development
                    if drift_correction == 1:
                        drift_detection_range_x = 5 # to be defined elsewhere
                        drift_detection_range_y = 3
                        drift_detection_results_array = numpy.zeros((1 + 2* drift_detection_range_x, 1 + 2* drift_detection_range_y))

                        array_zero_blur = gaussian_filter(array_zero, sigma=2)

                        print(i)#, first)
                        if i == first:
                            zero_first = numpy.array(array_zero_blur)
                            zero_first = numpy.reshape(zero_first, -1)
                        else:

                            x = -drift_detection_range_x
                            #print(x)
                            while x < drift_detection_range_x + 1:

                                y = -drift_detection_range_y
                                #print(y)
                                while y < drift_detection_range_y + 1:

                                    array_zero_shifted = ndimage.shift(
                                        numpy.single(numpy.array(array_zero_blur)), [y,x], order=3, mode='nearest', prefilter=True)
                                    array_zero_shifted = numpy.asarray(array_zero_shifted)
                                    array_zero_shifted = numpy.reshape(array_zero_shifted, -1)
                                    result = pearsonr(array_zero_shifted, zero_first)
                                    #print(x, y, result[0])
                                    drift_detection_results_array[x+ drift_detection_range_x, y+ drift_detection_range_y] = result[0]

                                    y = y + 1
                                x = x + 1

                            #print(drift_detection_results_array)
                            ind = numpy.argmax(drift_detection_results_array)
                            ind_2d = numpy.unravel_index(ind, drift_detection_results_array.shape)
                            shift_2d = (ind_2d[0]-drift_detection_range_x, ind_2d[1]-drift_detection_range_y)
                            print(shift_2d)



                if self.Abort_and_reconstruct_now.isChecked() == True:
                    break
            if self.Abort_and_reconstruct_now.isChecked() == True:
                break

            filename = path_in + namepart + str(n).zfill(4) + filetype
            filename_ = path_in + namepart + str(n + 1).zfill(4) + filetype

            while os.path.exists(filename_) != True:
                time.sleep(1)
                print('Waiting for next Projection:', filename)
                self.logbook.append(
                    strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Waiting for next Projection:' + filename)
                QtCore.QCoreApplication.processEvents()
                time.sleep(0.02)
                if self.Abort_and_reconstruct_now.isChecked() == True:
                    break

            if self.Abort_and_reconstruct_now.isChecked() == True:
                break

            # LOAD FILE
            print('Loading Projection ', filename)
            self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Loading Projection ' + filename)
            ima = Image.open(filename)

            if self.transpose == True:
                ima = ima.transpose(Image.TRANSPOSE)


            # NORMALIZE
            ima = numpy.single(numpy.array(ima))
            proj_sub = numpy.subtract(ima, DF)
            FF_sub = numpy.subtract(numpy.array(FF_avg), numpy.array(DF))
            im_normalized = numpy.divide(proj_sub, FF_sub)
            im_normalized = numpy.clip(im_normalized, 0.03, 5)
            im_normalized = im_normalized*16000
            im_normalized_16 = im_normalized.astype(numpy.uint16)

            #im_normalized = numpy.nan_to_num(im_normalized, copy=True, nan=1.0, posinf=1.0, neginf=1.0)


            # OPTICAL DISTORTION CORRECTION #
            # some image morphing based on the applied lens distortion correction could be implemented here

            # EXTEND
            if extend_FOV != 0:
                arr = tomopy.misc.morph.pad(im_normalized_16, axis=1, npad=round(2 * extend_FOV * im.size[0]), mode='edge',
                                            ncore=self.no_of_cores)
            else:
                arr = im_normalized_16

            arr = numpy.nan_to_num(arr, copy=True, nan=1.0, posinf=1.0, neginf=1.0)

            # ROTATE (TILT CORRECTION)
            if self.rotate != 0:
                arr = ndimage.rotate(arr, -(self.rotate/2), axes=[1, 0], reshape=False, output=None, order=3, mode='nearest', cval=0.0, prefilter=True)


            # SHIFT (SUB PIXEL SHIFT)

            if self.find_pixel_size_vertical == True:
                arr = ndimage.shift(numpy.single(numpy.array(arr)), [-(x_offset_list[i - first] / self.pixel_size), 0],
                                                          order=3, mode='nearest', prefilter=True)
            else:
                arr = ndimage.shift(numpy.single(numpy.array(arr)), [0, (x_offset_list[i - first] / self.pixel_size)],
                                                          order=3, mode='nearest', prefilter=True)


            arr16 = arr.astype(numpy.uint16)

            # SAVE NORMALIZED PROJECTIONS #
            if save_normalized == 1:
                if save_normalized_classic_order == 1 and number_of_sequences != 1:
                    filename3 = path_out_normalized + namepart + str(int( number_of_sequences * ((i-1) % sequence_size)  +  (theta_first_list[math.floor((i-1)/sequence_size)] - min(theta_first_list)) * number_of_sequences +1 )).zfill(4) + filetype             #INDEPENDENT_FIRST
                else:
                    filename3 = path_out_normalized + namepart + str(i).zfill(4) + filetype

                print('Writing Normalized Projection ', filename3)
                self.logbook.append(
                    strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Writing Normalized Projection ' + filename3)
                if extend_FOV != 0:
                    norm = Image.fromarray(arr16[:, round(2 * extend_FOV * im.size[0]): round((2 * extend_FOV + 1) * im.size[0])])
                else:
                    norm = Image.fromarray(arr16)
                norm.save(filename3)

            # LOGARITHM
            arr = tomopy.minus_log(arr16)   # here uint is necessarry!!!
            arr = (arr + 9.68)*1000         # conversion factor to uint
            #arr = numpy.clip(arr, 0.0, 5)
            arr = numpy.atleast_3d(arr)
            arr2 = numpy.swapaxes(arr, 0, 2)
            arr2 = numpy.swapaxes(arr2, 1, 2)

            if i == first:
                arra = numpy.zeros((number_of_sequences * sequence_size, arr2.shape[1], arr2.shape[2]), dtype=numpy.int16)
                arra[i - first, :, :] = numpy.uint16(arr2)

            else:
                arra[i - first, :, :] = numpy.uint16(arr2)
            print('Data Dimensions so far: ', i, ' of ', number_of_sequences * sequence_size, arra.shape)
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Data Dimensions so far: ' + str(i) + ' of ' + str(
                    number_of_sequences * sequence_size) + str(arra.shape))

            #print('arra:', arra.dtype)

            # RECONSTRUCT IF LAST FILE # ================================================================================== # RECONSTRUCT IF LAST FILE #
            if os.path.exists(path_in + namepart + str(n + 2).zfill(4) + filetype) != True or (
                    i % self.preview_frequency.value()) == 0:

                factor = (math.pi / 180)                            # Theta_list is in degr, tomopy wants rad
                angle_list = [(i+self.Offset_Angle.value()) * factor for i in theta_list]
                angle_list = angle_list[:i]
                #new_list_w_offset = new_list + factor * self.Offset_Angle.value()
                cor = self.COR_change.value() + round(2 * extend_FOV * im.size[0])
                slice = self.Preview_slice.value()
                if slice > im.size[1]:
                    slice = im.size[1]-1
                    self. Preview_slice.setValue(im.size[1]-1)

                center_list = [cor] * (i)
                arrar = arra[0:i - first + 1, slice - 2:slice + 1, :]

                if self.checkBox_phase.isChecked() == True:
                    print('Performing phase retrieval')
                    self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Performing phase retrieval' + str(arrar.shape))
                    arrar = tomopy.prep.phase.retrieve_phase(arrar, pixel_size = self.pixel_size/10000, dist = self.doubleSpinBox_distance.value(), energy = self.doubleSpinBox_Energy.value(), alpha = self.doubleSpinBox_alpha.value(), pad = True, ncore = self.no_of_cores, nchunk = None)


                print('Reconstructing Data', arrar.shape)
                self.logbook.append(
                    strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Reconstructing Data' + str(arrar.shape))


                if self.algorithm_list.currentText() == 'FBP_CUDA':
                    options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
                    slices = tomopy.recon(arrar, angle_list, center=center_list, algorithm=tomopy.astra, options=options, ncore=self.no_of_cores)

                else:
                    slices = tomopy.recon(arrar, angle_list, center=center_list, algorithm=self.algorithm_list.currentText(),
                                      filter_name=self.filter_list.currentText(), ncore=self.no_of_cores)

                slices = slices[:, round(1 * extend_FOV * im.size[0]): -round(1 * extend_FOV * im.size[0]), round(1 * extend_FOV * im.size[0]): -round(1 * extend_FOV * im.size[0])]
                slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)

                ima2 = slices[1,:,:]
                #ima3 = (ima2+100)*320

                self.min.setText(str(numpy.amin(ima2)))
                self.max.setText(str(numpy.amax(ima2)))

                print('Reconstructed Volume is', slices.shape)
                self.logbook.append(
                    strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Reconstructed Volume is' + str(slices.shape))
                filename2 = path_out_reconstructed + namepart + str(slice).zfill(4) + '_' + str(i).zfill(
                    4) + filetype
                print('Writing Reconstructed Slice:', filename2)
                self.logbook.append(
                    strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Writing Reconstructed Slice:' + filename2)

                if self.radioButton_16bit_integer.isChecked() == True:
                    ima3 = 65535 * (ima2 - self.int_low.value()) / (self.int_high.value() - self.int_low.value())
                    ima3 = numpy.clip(ima3, 1, 65534)
                    ima16 = ima3.astype(numpy.uint16)
                    if self.savePreviewOnDisk.isChecked() == True:
                        img = Image.fromarray(ima16)
                        img.save(filename2)
                        print('Reconstructed Slice has been written.')

                if self.radioButton_32bit_float.isChecked() == True:
                    if self.savePreviewOnDisk.isChecked() == True:
                        img = Image.fromarray(ima2)
                        img.save(filename2)
                        print('Reconstructed Slice has been written.')



                myarray = ima2 * self.BrightnessSlider.value()   # show preview     #something is wrong, it should be 8-bit?
                myarray = numpy.clip(myarray, 1, 255)
                yourQImage = qimage2ndarray.array2qimage(myarray)
                self.preview.setPixmap(QPixmap(yourQImage))
                print('Reconstructed Slice displayed')


            f = i
            i = i + 1
            n = n + 1

            if self.Abort_and_reconstruct_now.isChecked() == True:
                break

        QtWidgets.QApplication.processEvents()

        # WAITING LOOP BEFORE COMPLETE VOLUME RECONSTRUCTION # VERY LAGGY
        """
        while self.checkBox_reconstruct_at_end.isChecked() == False:

            self.checkBox_reconstruct_at_end.setText('Reconstruct Volume now')

            angle_list = [(i + self.Offset_Angle.value()) * factor for i in theta_list]
            angle_list = angle_list[:i]
            # new_list_w_offset = new_list + factor * self.Offset_Angle.value()
            cor = self.COR_change.value() + round(2 * extend_FOV * im.size[0])
            slice = self.Preview_slice.value()
            if slice > im.size[1]:
                slice = im.size[1] - 1
                self.Preview_slice.setValue(im.size[1] - 1)

            center_list = [cor] * (i)
            arrar = arra[0:i - first + 1, slice - 2:slice + 1, :]

            QtWidgets.QApplication.processEvents()

            if self.checkBox_phase.isChecked() == True:
                print('Performing phase retrieval')
                self.logbook.append(
                    strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Performing phase retrieval' + str(arrar.shape))
                arrar = tomopy.prep.phase.retrieve_phase(arrar, pixel_size=self.pixel_size / 10000,
                                                         dist=self.doubleSpinBox_distance.value(),
                                                         energy=self.doubleSpinBox_Energy.value(),
                                                         alpha=self.doubleSpinBox_alpha.value(), pad=True,
                                                         ncore=self.no_of_cores, nchunk=None)

            QtWidgets.QApplication.processEvents()

            print('Reconstructing Data', arrar.shape)
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Reconstructing Data' + str(arrar.shape))

#           slices = tomopy.recon(arrar, angle_list, center=center_list, algorithm=self.algorithm_list.currentText(),
#                                  filter_name=self.filter_list.currentText(), ncore=self.no_of_cores)

            if self.algorithm_list.currentText() == 'FBP_CUDA':
                options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
                slices = tomopy.recon(arrar, angle_list, center=center_list, algorithm=tomopy.astra, options=options, ncore=self.no_of_cores)

            else:
                slices = tomopy.recon(arrar, angle_list, center=center_list, algorithm=self.algorithm_list.currentText(), filter_name=self.filter_list.currentText(), ncore=self.no_of_cores)



            slices = slices[:, round(1 * extend_FOV * im.size[0]): -round(1 * extend_FOV * im.size[0]),
                     round(1 * extend_FOV * im.size[0]): -round(1 * extend_FOV * im.size[0])]
            slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)

            ima2 = slices[1, :, :]
            ima3 = (ima2 + 100) * 320
            ima3 = numpy.clip(ima3, 1, 65534)
            ima16 = ima3.astype(numpy.uint16)

            QtWidgets.QApplication.processEvents()

            print('Reconstructed Volume is', slices.shape)
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Reconstructed Volume is' + str(slices.shape))
            filename2 = path_out_reconstructed + namepart + str(slice).zfill(4) + '_' + str(i).zfill(
                4) + filetype

            print('Writing Reconstructed Slice:', filename2)
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Writing Reconstructed Slice:' + filename2)

            if self.savePreviewOnDisk.isChecked() == True:
                img = Image.fromarray(ima16)
                img.save(filename2)
                print('Reconstructed Slice has been written.')

            myarray = (ima16 - 31000) * 0.005 * self.BrightnessSlider.value()  # show preview
            myarray = numpy.clip(myarray, 1, 255)
            yourQImage = qimage2ndarray.array2qimage(myarray)
            self.preview.setPixmap(QPixmap(yourQImage))
            print('Reconstructed Slice displayed')

            QtWidgets.QApplication.processEvents()
        """


        # APPLYING MOVEMENT CORRECTION # =============================================================================== # APPLYING MOVEMENT CORRECTION #

        if self.apply_movement_correction.isChecked() == True:

            sinogram = arra[:, slice , :]
            print(sinogram.shape)
            transposed_sinogram = numpy.transpose(sinogram, axes=[1, 0])
            print(transposed_sinogram.shape)
            print('Applying movement correction')

            max_theta = numpy.max(theta_list)
            print('path_out',path_out, 'namepart', namepart, 'number of seq', number_of_sequences, 'seq size', sequence_size, 'cor', cor,'max theta', max_theta, 'theta first list', theta_first_list)
            time.sleep(3)


            main_Movement_corrector = Movement_corrector(transposed_sinogram, path_out, namepart, number_of_sequences, sequence_size, cor, max_theta, theta_first_list)
            print('main_Movement_corrector = Movement_corrector(transposed_sinogram, path_out, namepart, number_of_sequences, sequence_size, cor, max_theta, theta_first_list)')

            main_Movement_corrector.show()
            print('main_Movement_corrector.show()')

            while main_Movement_corrector.movement_correction_running == 1:
                time.sleep(0.1)
                QtWidgets.QApplication.processEvents()
                #print('waiting for movement correction running = ', main_Movement_corrector.movement_correction_running)

            movement_corrector_results = main_Movement_corrector.movement_corrector_results
            print('movement_corrector_results = main_Movement_corrector.movement_corrector_results')
            checkBox_save_shifted_projections = main_Movement_corrector.checkBox_save_shifted_projections
            print('checkBox_save_shifted_projections = main_Movement_corrector.checkBox_save_shifted_projections')
            path_out_corr = main_Movement_corrector.path_out_corr
            print('path_out_corr = main_Movement_corrector.path_out_corr')


            print('movement_corrector finished - back in main CT-Preview-Reco')
            print(movement_corrector_results)

            if numpy.max(movement_corrector_results) != 0:
                print('Applying shift correction')
                n = 0
                while n < number_of_projections:
                    print(n, int(number_of_sequences * ((n) % sequence_size)  +  (theta_first_list[math.floor((n)/sequence_size)] - min(theta_first_list)) * number_of_sequences ))
                    self.progressBar_Sequence.setValue((n + 1) * 100 / number_of_projections)
                    QtWidgets.QApplication.processEvents()

                    temp = ndimage.shift(arra[n,:,:], (0, movement_corrector_results[int(number_of_sequences * ((n) % sequence_size)  +  (theta_first_list[math.floor((n)/sequence_size)] - min(theta_first_list)) * number_of_sequences )]), mode='nearest', prefilter=True)
                    temp16 = temp.astype(numpy.uint16)
                    arra[n, :, :] = temp16

                    n = n + 1

                print('arra:', arra.dtype)

                if checkBox_save_shifted_projections == True:

                    if os.path.isdir(path_out_corr + '/Normalized_Projections_Shift_Corrected') is False:
                        os.mkdir(path_out_corr + '/Normalized_Projections_Shift_Corrected')
                        print('creating Normalized_Projections_Shift_Corrected', path_out_corr,
                              '/Normalized_Projections_Shift_Corrected')

                    i = 1
                    while i < number_of_projections + 1:
                        self.progressBar_Total.setValue((i + 1) * 100 / number_of_projections)
                        QtWidgets.QApplication.processEvents()

                        if save_normalized_classic_order == 1:
                            filename3 = path_out_corr + '/Normalized_Projections_Shift_Corrected' + namepart + str(int( number_of_sequences * ((i-1) % sequence_size)  +  (theta_first_list[math.floor((i-1)/sequence_size)] - min(theta_first_list)) * number_of_sequences +1 )).zfill(4) + filetype             #INDEPENDENT_FIRST
                        else:
                            filename3 = path_out_corr + '/Normalized_Projections_Shift_Corrected' + namepart + str(i).zfill(4) + filetype

                        print('Writing Normalized Projection ', filename3)
                        self.logbook.append(
                            strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Writing shifted Projection ' + filename3)
                        if extend_FOV != 0:
                            yarra = (arra[i - 1, :, round(2 * extend_FOV * im.size[0]): round((2 * extend_FOV + 1) * im.size[0])] * 0.001) - 9.68
                            yarra = numpy.exp(numpy.negative(yarra))
                            yarra16 = yarra.astype(numpy.uint16)
                            norm_proj_shift_corr = Image.fromarray(yarra16)
                        else:
                            yarra = (arra[i - 1, :, :] * 0.001) - 9.68
                            yarra = numpy.exp(numpy.negative(yarra))
                            yarra16 = yarra.astype(numpy.uint16)
                            norm_proj_shift_corr = Image.fromarray(yarra16)

                        norm_proj_shift_corr.save(filename3)
                        i = i + 1

        QtWidgets.QApplication.processEvents()



        # RECONSTRUCT COMPLETE VOLUME # ================================================================================ # RECONSTRUCT COMPLETE VOLUME #

        #new_list = [i * factor for i in theta_list]
        angle_list = [(i + self.Offset_Angle.value()) * factor for i in theta_list]
        #new_list_w_offset = new_list + factor * self.Offset_Angle.value()

        cor = self.COR_change.value() + round(2 * extend_FOV * im.size[0])
        center_list = [cor] * (i)

        if self.checkBox_reconstruct_at_end.isChecked() == True:

            arra = arra[: f - first+1,:,:]
            print('arra:', arra.dtype)
            print('checking conditions for adv. ringfilter')

            if self.advanced_ringfilter.isChecked() == True and self.Abort_and_reconstruct_now.isChecked() == False:
                print('Applying advanced ring filter')
                arratwo = numpy.copy(arra)
                arratwo.fill(0.0)
                print('arra-shape', arra.shape)
                print('arratwo-shape', arratwo.shape)


                m = 0
                while (m < sequence_size * number_of_sequences):
                    print('index', m,' result', int( number_of_sequences * ((m-1) % sequence_size)  +  (theta_first_list[math.floor((m-1)/sequence_size)] - min(theta_first_list)) * number_of_sequences), ' last', f-first)
                    temp = arra[int(m), :, :]
                    arratwo[int( number_of_sequences * ((m-1) % sequence_size)  +  (theta_first_list[math.floor((m-1)/sequence_size)] - min(theta_first_list)) * number_of_sequences), :, :] = temp
                    m = m + 1

                filename_ring_before = path_in + namepart + '_original_sinogram' + filetype
                img = Image.fromarray(arra[:, 11, :])
                img.save(filename_ring_before)

                #arrathree = ndimage.median_filter(arratwo, footprint = [[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]]], mode ='nearest')
                print('Lets start filtering')
                arrathree = numpy.copy(arratwo)
                deviation = 5           # THRESHOLD MEDIAN FILTER
                n = 0
                while n < arratwo.shape[1]:
                    print(n,arratwo.shape)
                    imathree = ndimage.median_filter(arratwo[:,n,:], footprint=[[0, 1, 0], [0, 1, 0], [0, 1, 0]], mode='nearest')
                    print('median successful')
                    divided = numpy.divide(arratwo[:,n,:], imathree)
                    print('divide successful')
                    divided = numpy.nan_to_num(divided, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
                    a = divided < 100 / (100 + deviation)   # True False array
                    b = divided > (100 + deviation) / 100   # True False array
                    c = a.astype(int) + b.astype(int)       # convert to 1 and 0
                    d = numpy.clip(c, 0, 1)
                    e = -d + 1
                    g = d * imathree + e * arratwo[:,n,:]
                    print('filling into array')
                    arrathree[:,n,:] = g
                    n = n + 1
                print('filtering done. Shape arrathree: ', arrathree.shape)
                print('Shape arra: ', arra.shape)


                filename_ring_after = path_in + namepart + '_after_ringfilter' + filetype
                img = Image.fromarray(arrathree[:, 11, :])
                img.save(filename_ring_after)

                print('starting to reorder again')
                print('arra and arrathree shape',arra.shape, arrathree.shape)
                m = 0
                while (m < f - first+1):
                    print(m)
                    arra[int(m),:,:] = arrathree[int( number_of_sequences * ((m-1) % sequence_size)  +  (theta_first_list[math.floor((m-1)/sequence_size)] - min(theta_first_list)) * number_of_sequences),:,:]
                    m = m + 1
                print('Advanced ringfilter finished')

            print('Advanced ringfilter passed')

            if self.checkBox_phase.isChecked() == True:
                print('Performing phase retrieval')
                self.logbook.append(
                    strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Performing phase retrieval' + str(arra.shape))
                arra = tomopy.prep.phase.retrieve_phase(arra, pixel_size = self.pixel_size / 10000,
                                                         dist=self.doubleSpinBox_distance.value(),
                                                         energy=self.doubleSpinBox_Energy.value(),
                                                         alpha=self.doubleSpinBox_alpha.value(), pad=True, ncore=self.no_of_cores,
                                                         nchunk=None)


            i = 0
            while (i < math.ceil(arra.shape[1] / self.block_size)):

                print('Reconstructing block', i + 1, 'of', math.ceil(arra.shape[1] / self.block_size))
                self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Reconstructing block ' + str(
                    i + 1) + ' of ' + str(math.ceil(arra.shape[1] / self.block_size)))

                # RECONSTRUCTING # =====================================================================================

                if self.algorithm_list.currentText() == 'FBP_CUDA':
                    options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
                    slices = tomopy.recon(arra[:, i * self.block_size: (i + 1) * self.block_size, :], angle_list, center=center_list, algorithm=tomopy.astra, options=options, ncore=self.no_of_cores)

                else:
                    slices = tomopy.recon(arra[:, i * self.block_size: (i + 1) * self.block_size, :], angle_list, center=center_list, algorithm=self.algorithm_list.currentText(),
                                      filter_name=self.filter_list.currentText(), ncore=self.no_of_cores)


#                slices = tomopy.recon(arra[:, i * self.block_size: (i + 1) * self.block_size, :], angle_list,
#                                      center=center_list, algorithm=self.algorithm_list.currentText(),
#                                      filter_name=self.filter_list.currentText(), ncore=self.no_of_cores)
                slices = slices[:, round(1 * extend_FOV * im.size[0]): -round(1 * extend_FOV * im.size[0]), round(1 * extend_FOV * im.size[0]): -round(1 * extend_FOV * im.size[0])]
                slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)







                if self.radioButton_16bit_integer.isChecked() == True:
                    ima3 = 65535 * (slices - self.int_low.value()) / (self.int_high.value() - self.int_low.value())
                    ima3 = numpy.clip(ima3, 1, 65534)
                    slices_save = ima3.astype(numpy.uint16)


                if self.radioButton_32bit_float.isChecked() == True:
                    slices_save = slices



                print('Reconstructed Volume is', slices_save.shape)
                self.logbook.append(
                    strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Reconstructed Volume is' + str(slices_save.shape))


                a = 1
                while (a < self.block_size + 1) and (a < slices_save.shape[0] + 1):
                    filename2 = path_out_reconstructed_full + namepart + str(
                        a + volume_begin + i * self.block_size).zfill(4) + filetype
                    print('Writing Reconstructed Slices:', filename2)
                    self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ",
                                                 localtime()) + 'Writing Reconstructed Slices:' + filename2)

                    slice_save = slices_save[a - 1, :, :]
                    img = Image.fromarray(slice_save)

                    img.save(filename2)
                    self.progressBar_Reconstruction.setValue((a + (i * self.block_size)) * 100 / arra.shape[1])
                    QtCore.QCoreApplication.processEvents()
                    time.sleep(0.02)

                    a = a + 1

                i = i + 1


        # DIFFERENCE IMAGE AT 0, 90 AND 180° # ============================================================================ # DIFFERENCE IMAGE AT 0, 90 AND 180° #

        if self.Abort_and_reconstruct_now.isChecked() != True:

            zero_deg_offset = 0
            if zero_deg_proj == 2:
                zero_deg_offset = number_of_sequences

            filename1 = path_in + namepart + str(
                number_of_sequences * sequence_size + (number_of_sequences + 1) * FF_sequence_size + 4 + zero_deg_offset) .zfill(
                4) + filetype
            filename2 = path_in + namepart + str(
                number_of_sequences * sequence_size + (number_of_sequences + 1) * FF_sequence_size + 5 + zero_deg_offset).zfill(
                4) + filetype
            filename3 = path_in + namepart + str(
                number_of_sequences * sequence_size + (number_of_sequences + 1) * FF_sequence_size + 6 + zero_deg_offset).zfill(
                4) + filetype
            filename4 = path_in + namepart + str(
                number_of_sequences * sequence_size + (number_of_sequences + 1) * FF_sequence_size + 3 + zero_deg_offset).zfill(
                4) + filetype

            while os.path.exists(filename3) != True:
                time.sleep(1)
                print('waiting for last file:', filename3)
                self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'waiting for last file:' + filename3)
            time.sleep(5)

            eim_000deg = Image.open(filename1)
            eim_090deg = Image.open(filename2)
            eim_180deg = Image.open(filename3)
            FF = Image.open(filename4)

            if self.transpose == True:
                eim_000deg = eim_000deg.transpose(Image.TRANSPOSE)
                eim_090deg = eim_090deg.transpose(Image.TRANSPOSE)
                eim_180deg = eim_180deg.transpose(Image.TRANSPOSE)
                FF = FF.transpose(Image.TRANSPOSE)


            eim_000deg = numpy.single(numpy.array(eim_000deg))
            eim_090deg = numpy.single(numpy.array(eim_090deg))
            eim_180deg = numpy.single(numpy.array(eim_180deg))
            eproj_000_sub = numpy.subtract(eim_000deg, DF)
            eproj_090_sub = numpy.subtract(eim_090deg, DF)
            eproj_180_sub = numpy.subtract(eim_180deg, DF)
            FF_sub = numpy.subtract(numpy.array(FF), numpy.array(DF))
            eim_000_normalized = numpy.divide(eproj_000_sub, FF_sub)
            eim_090_normalized = numpy.divide(eproj_090_sub, FF_sub)
            eim_180_normalized = numpy.divide(eproj_180_sub, FF_sub)
            div_000_normalized = numpy.divide(eim_000_normalized, im_000_normalized)
            div_090_normalized = numpy.divide(eim_090_normalized, im_090_normalized)
            div_180_normalized = numpy.divide(eim_180_normalized, im_180_normalized)

            filename_e_000 = path_out_changes + '/Paulchen_Normalized' + namepart + 'end_000_deg' + filetype
            print('End Projection at 0°:', filename_e_000)
            self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'End Projection at 0°:' + filename_e_000)
            img = Image.fromarray(eim_000_normalized)
            img.save(filename_e_000)
            filename_e_090 = path_out_changes + '/Paulchen_Normalized' + namepart + 'end_090_deg' + filetype
            print('End Projection at 90°:', filename_e_090)
            self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'End Projection at 90°:' + filename_e_090)
            img = Image.fromarray(eim_090_normalized)
            img.save(filename_e_090)
            filename_e_180 = path_out_changes + '/Paulchen_Normalized' + namepart + 'end_180_deg' + filetype
            print('End Projection at 180°:', filename_e_180)
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'End Projection at 180°:' + filename_e_180)
            img = Image.fromarray(eim_180_normalized)
            img.save(filename_e_180)

            filename_000 = path_out_changes + '/Paulchen_Divided' + namepart + 'div_000_deg' + filetype
            print('Difference in Projection at 0°:', filename_000)
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Difference in Projection at 0°:' + filename_000)
            img = Image.fromarray(div_000_normalized)
            img.save(filename_000)
            filename_090 = path_out_changes + '/Paulchen_Divided' + namepart + 'div_090_deg' + filetype
            print('Difference in Projection at 90°:', filename_090)
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Difference in Projection at 90°:' + filename_090)
            img = Image.fromarray(div_090_normalized)
            img.save(filename_090)
            filename_180 = path_out_changes + '/Paulchen_Divided' + namepart + 'div_180_deg' + filetype
            print('Difference in Projection at 180°:', filename_180)
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Difference in Projection at 180°:' + filename_180)
            img = Image.fromarray(div_180_normalized)
            img.save(filename_180)

        print('Done!')
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Done!')
        protocol = self.logbook.toPlainText()
        print(len(protocol), ' signs saved in protocol')
        text_file = open(self.file_name_protocol, "wt")
        z = text_file.write(protocol)
        text_file.close()



        sys.exit(0)


    def closeEvent(self, event: QCloseEvent):
        print('Aborted')
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Aborted!')
        protocol = self.logbook.toPlainText()
        print(len(protocol), ' signs saved in protocol')
        text_file = open(self.file_name_protocol, "wt")
        z = text_file.write(protocol)
        text_file.close()
        sys.exit(app.exec_())



if __name__ == '__main__':


    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    main = CT_preview(0,0)
    main.show()
    mains
    sys.exit(app.exec_())


#print('Done')
