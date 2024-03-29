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


Ui_CT_previewWindow, QCOR_previewWindow = loadUiType('CT-preview_phase.ui')  # GUI vom Hauptfenster


class CT_preview(Ui_CT_previewWindow, QCOR_previewWindow):

    def __init__(self, COR, rotate, pixel_size, path_klick, path_out, block_size, dark_field_value, no_of_cores, checkBox_save_normalized, checkBox_classic_order, transpose, find_pixel_size_vertical):#, algorithm, filter):
        super(CT_preview, self).__init__()
        self.setupUi(self)

        self.setWindowTitle('Interlaced-CT Normalization Only')

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

        self.Start.clicked.connect(self.mains)

        self.algorithm_list.setEnabled(False)
        self.filter_list.setEnabled(False)
        self.COR_change.setEnabled(False)
        self.Preview_slice.setEnabled(False)
        self.preview_frequency.setEnabled(False)
        self.savePreviewOnDisk.setEnabled(False)
        self.checkBox_reconstruct_at_end.setEnabled(False)
        self.advanced_ringfilter.setEnabled(False)
        self.Abort_and_reconstruct_now.setEnabled(False)

        self.tabWidget.setCurrentIndex(1)

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

        slice = self.Preview_slice.value()              # Preview Slice                                                                     # Whats wrong here? #
        #double_grid_size = 0                            # =1 doubles the saved reconstructed area; essential for full extended field of view scans (half sided 360 degree scans)

        if self.checkBox_save_normalized == True:
            save_normalized = 1                         # save normalized projections?
        else:
            save_normalized = 0

        if self.checkBox_classic_order == True:
            save_normalized_classic_order = 1           # sorts the normalized projections to steadily rising angles !!! SET THIS 0 FOR CLASSIC-CT !!!
        else:
            save_normalized_classic_order = 0

        #preview_frequency = 10                          # preview reconstruction on idle and every 5th, 10th, 20th or 50th projection
        #volume_begin = 0                                # reconstruct these slices in the end


        # DEFINING PATHS # ==================================================================================================== # DEFINING PATHS #

        root = Tkinter.Tk()

        path_out = self.path_out

        path_klick = self.path_klick
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_klick)
        htap = self.path_klick[::-1]
        path_in = self.path_klick[0: len(htap) - htap.find('/') - 1: 1]
        namepart = self.path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 5: 1]
        #counter = self.path_klick[len(htap) - htap.find('.') - 5: len(htap) - htap.find('.') - 1:1]
        filetype = self.path_klick[len(htap) - htap.find('.') - 1: len(htap):1]


        print(path_out)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_out)

        path_lists = path_in
        print(path_lists)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_lists)
        root.withdraw()

        #path_out_reconstructed = path_out + '/Reconstructed_Preview'
        #path_out_reconstructed_full = path_out + '/Reconstructed_Volume'
        path_out_normalized = path_out + '/Normalized_Projections'
        path_out_changes = path_out + '/Changes'

        #print(path_out_reconstructed)
        #print(path_out_reconstructed_full)
        print(path_out_normalized)
        print(path_out_changes)
        #self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_out_reconstructed)
        #self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_out_reconstructed_full)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_out_normalized)
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + path_out_changes)

        #if os.path.isdir(path_out_reconstructed) is False:
        #    os.mkdir(path_out_reconstructed)
        #if os.path.isdir(path_out_reconstructed_full) is False:
        #    os.mkdir(path_out_reconstructed_full)
        if save_normalized == 1:
            if os.path.isdir(path_out_normalized) is False:
                os.mkdir(path_out_normalized)
        if os.path.isdir(path_out_changes) is False:
            os.mkdir(path_out_changes)



        first = 1

        self.file_name_protocol = path_out + '/' + 'reconstruction_protocol.txt'

        # READ THETA AND X_OFFSET-LIST # ====================================================================================== # READ THETA AND X_OFFSET-LIST #

        file_name_theta = path_lists + '/theta_list.txt'
        #theta_list = numpy.genfromtxt(file_name_theta)
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



        # OPEN AND NORMALIZE 0, 90 AND 180° # ============================================================================= # OPEN AND NORMALIZE 0, 90 AND 180° #

        #print(self.algorithm, 'algorithm chosen')
        #print(self.filter, 'filter chosen')
        #self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + str(self.algorithm) + 'algorithm chosen')
        #self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + str(self.filter) + 'filter chosen')
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
        im = im_000deg
        im_090deg = Image.open(filename2)
        im_180deg = Image.open(filename3)
        FF = Image.open(filename4)
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

        filename_b_000 = path_out_changes + namepart + 'beginning_000_deg' + filetype
        print('Beginning Projection at 0°:', filename_b_000)
        self.logbook.append(
            strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Beginning Projection at 0°:' + filename_b_000)
        img = Image.fromarray(im_000_normalized)
        img.save(filename_b_000)

        filename_b_090 = path_out_changes + namepart + 'beginning_090_deg' + filetype
        print('Beginning Projection at 90°:', filename_b_090)
        self.logbook.append(
            strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Beginning Projection at 90°:' + filename_b_090)
        img = Image.fromarray(im_090_normalized)
        img.save(filename_b_090)

        filename_b_180 = path_out_changes + namepart + 'beginning_180_deg' + filetype
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

        if slice > im.size[1] - 1:
            self.Preview_slice.setValue(round(im.size[1] / 2))
            print('Slice out of bound! Slice set to', round(im.size[1] / 2))
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + ' Slice out of bound! Slice set to ' + str(
                    round(im.size[1] / 2)))
            time.sleep(0.5)


        extend_FOV = (abs(self.COR - im.size[0]/2))/ (1 * im.size[0])     # extend field of view (FOV), 0.0 no extension, 0.5 half extension to both sides (for half sided 360 degree scan!!!)
        print('extend_FOV ', extend_FOV)




        # FOLLOW AND READ DATA AND RECONSTRUCT # ===========================================================================# FOLLOW AND READ DATA AND RECONSTRUCT #

        print('used cor:', self.COR)

        FF0 = numpy.ones((FF_sequence_size, im.size[1], im.size[0]), numpy.float32)
        print('stack size: number of images', last - first + 1, '; Y =', im.size[1], '; X =', im.size[0])
        self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + ' stack size: number of images ' + str(
            last - first + 1) + '; Y =' + str(im.size[1]) + '; X =' + str(im.size[0]))


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
                    filename_zero = path_out_changes + namepart + str(i - first).zfill(4) + filetype
                    print('Loading Zero Degree Projection ', filename_zero_load)
                    self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Loading Zero Degree Projection ' + filename_zero_load)
                    im_zero = Image.open(filename_zero_load)
                    im_zero = numpy.single(numpy.array(im_zero))
                    im_zero_sub = numpy.subtract(im_zero, DF)
                    FF_sub = numpy.subtract(numpy.array(FF_avg), numpy.array(DF))
                    im_zero_normalized = numpy.divide(im_zero_sub, FF_sub)

                    im_zero_normalized = numpy.nan_to_num(im_zero_normalized, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
                    im_zero_normalized = ndimage.shift(numpy.single(numpy.array(im_zero_normalized)), [0, (x_offset_list[i - first] / self.pixel_size)],
                                        order=3, mode='nearest', prefilter=True)

                    print('writing Zero Degree Projection ', filename_zero)
                    im_zero_normalized = Image.fromarray(im_zero_normalized)
                    im_zero_normalized.save(filename_zero)
                    n = n + 1



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


            # NORMALIZE
            ima = numpy.single(numpy.array(ima))
            proj_sub = numpy.subtract(ima, DF)

            proj_sub = numpy.nan_to_num(proj_sub, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
            proj_sub = numpy.clip(proj_sub, 1, 65534)

            FF_sub = numpy.subtract(numpy.array(FF_avg), numpy.array(DF))

            FF_sub = numpy.nan_to_num(FF_sub, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
            FF_sub = numpy.clip(FF_sub, 1, 65534)

            im_normalized = numpy.divide(proj_sub, FF_sub)
            im_normalized = numpy.nan_to_num(im_normalized, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
            im_normalized = numpy.clip(im_normalized, 0.01, 5)


            # EXTEND
            if extend_FOV != 0:
                arr = tomopy.misc.morph.pad(im_normalized, axis=1, npad=round(2 * extend_FOV * im.size[0]), mode='edge',
                                            ncore=self.no_of_cores)
            else:
                arr = im_normalized


            # ROTATE (TILT CORRECTION)
            arr = numpy.nan_to_num(arr, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
            if self.rotate != 0:
                arr = ndimage.rotate(arr, -(self.rotate/2), axes=[1, 0], reshape=False, output=None, order=3, mode='nearest', cval=0.0, prefilter=True)


            # SHIFT (SUB PIXEL SHIFT)
            arr = ndimage.shift(numpy.single(numpy.array(arr)), [0, (x_offset_list[i - first] / self.pixel_size)],
                                                          order=3, mode='nearest', prefilter=True)


            arr = arr*16000
            arr16 = arr.astype(numpy.uint16)

            # SAVE NORMALIZED PROJECTIONS #
            if save_normalized == 1:
                if save_normalized_classic_order == 1:
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


            i = i + 1
            n = n + 1

            if self.Abort_and_reconstruct_now.isChecked() == True:
                break


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

            filename_e_000 = path_out_changes + namepart + 'end_000_deg' + filetype
            print('End Projection at 0°:', filename_e_000)
            self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'End Projection at 0°:' + filename_e_000)
            img = Image.fromarray(eim_000_normalized)
            img.save(filename_e_000)
            filename_e_090 = path_out_changes + namepart + 'end_090_deg' + filetype
            print('End Projection at 90°:', filename_e_090)
            self.logbook.append(strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'End Projection at 90°:' + filename_e_090)
            img = Image.fromarray(eim_090_normalized)
            img.save(filename_e_090)
            filename_e_180 = path_out_changes + namepart + 'end_180_deg' + filetype
            print('End Projection at 180°:', filename_e_180)
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'End Projection at 180°:' + filename_e_180)
            img = Image.fromarray(eim_180_normalized)
            img.save(filename_e_180)

            filename_000 = path_out_changes + namepart + 'div_000_deg' + filetype
            print('Difference in Projection at 0°:', filename_000)
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Difference in Projection at 0°:' + filename_000)
            img = Image.fromarray(div_000_normalized)
            img.save(filename_000)
            filename_090 = path_out_changes + namepart + 'div_090_deg' + filetype
            print('Difference in Projection at 90°:', filename_090)
            self.logbook.append(
                strftime("%Y_%m_%d %H:%M:%S ", localtime()) + 'Difference in Projection at 90°:' + filename_090)
            img = Image.fromarray(div_090_normalized)
            img.save(filename_090)
            filename_180 = path_out_changes + namepart + 'div_180_deg' + filetype
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



