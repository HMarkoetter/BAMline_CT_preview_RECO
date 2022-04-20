# -*- coding: utf-8 -*-
version =  "Version 2022.04.20 a"

from PyQt5 import QtCore, QtGui, QtWidgets
#from pyqtgraph import PlotWidget
import pyqtgraph as pg
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import numpy as np
import math
import time
import csv
from epics import caget, caput, camonitor
import os
import sys

pg.setConfigOption('background', 'w')   # Plothintergrund weiß (2D)
pg.setConfigOptions(antialias=True)     # Enable antialiasing for prettier plots

Ui_GeneratorWindow, QGeneratorWindow = loadUiType('CT_helper.ui')  # GUI vom Hauptfenster

#print("Let's go")


class CT_helper(Ui_GeneratorWindow, QGeneratorWindow):

    def __init__(self):

        super(CT_helper, self).__init__()
        self.setupUi(self)

        self.Sequence_size.valueChanged.connect(self.generate_list)
        self.box_No_of_FFs.valueChanged.connect(self.generate_list)
        self.tomo_pos.valueChanged.connect(self.generate_list)
        self.zero_deg_checkBox.stateChanged.connect(self.generate_list)
        self.FF_pos.valueChanged.connect(self.generate_list)
        self.box_lateral_shift.valueChanged.connect(self.generate_list)
        index = self.No_of_sequences.findText('16', QtCore.Qt.MatchFixedString)
        self.No_of_sequences.setCurrentIndex(index)
        self.No_of_sequences.currentIndexChanged.connect(self.generate_list)
        self.angular_range.valueChanged.connect(self.generate_list)
        self.save.clicked.connect(self.save_files)

        self.Sequence_size_2.valueChanged.connect(self.generate_list_classic)
        self.No_of_sequences_2.valueChanged.connect(self.generate_list_classic)
        self.box_No_of_FFs_2.valueChanged.connect(self.generate_list_classic)
        self.angular_range_2.valueChanged.connect(self.generate_list_classic)
        self.tomo_pos_2.valueChanged.connect(self.generate_list_classic)
        self.FF_pos_2.valueChanged.connect(self.generate_list_classic)
        self.save_2.clicked.connect(self.save_files_classic)

        self.Sequence_size_3.valueChanged.connect(self.generate_list_refraction)
        self.box_No_of_FFs_3.valueChanged.connect(self.generate_list_refraction)
        self.tomo_pos_3.valueChanged.connect(self.generate_list_refraction)
        self.zero_deg_checkBox_3.stateChanged.connect(self.generate_list_refraction)
        self.FF_pos_3.valueChanged.connect(self.generate_list_refraction)
        self.box_lateral_shift_3.valueChanged.connect(self.generate_list_refraction)
        index = self.No_of_sequences_3.findText('16', QtCore.Qt.MatchFixedString)
        self.No_of_sequences_3.setCurrentIndex(index)
        self.No_of_sequences_3.currentIndexChanged.connect(self.generate_list_refraction)
        self.angular_range_3.valueChanged.connect(self.generate_list_refraction)
        self.save_3.clicked.connect(self.save_files_refraction)

        self.getTomoPosition.clicked.connect(self.get_Tomo_pos)
        self.getTomoPosition_2.clicked.connect(self.get_Tomo_pos_2)
        self.getTomoPosition_3.clicked.connect(self.get_Tomo_pos_3)
        self.getFFPosition.clicked.connect(self.get_FF_pos)
        self.getFFPosition_2.clicked.connect(self.get_FF_pos_2)
        self.getFFPosition_3.clicked.connect(self.get_FF_pos_3)

        self.on_the_fly_comboBox.currentIndexChanged.connect(self.calculate_speed)
        self.On_the_fly_nr_projections.valueChanged.connect(self.calculate_speed)
        self.on_the_fly_exp_time.valueChanged.connect(self.calculate_speed)

        #print('init')
        self.generate_list()
        self.generate_list_classic()
        self.generate_list_refraction()


        # Calculate speeds for on-the-fly CT
    def calculate_speed(self):
        self.rot_speed.setValue(int((self.on_the_fly_comboBox.currentText()))/(self.on_the_fly_exp_time.value()*self.On_the_fly_nr_projections.value()))
        self.rot_step.setValue(int((self.on_the_fly_comboBox.currentText()))/(self.On_the_fly_nr_projections.value()))
        self.duration.setValue((self.on_the_fly_exp_time.value()*self.On_the_fly_nr_projections.value())/60)


    def generate_list(self):

        # GET PARAMETERS #
        number_of_sequences = int((self.No_of_sequences.currentText()))
        sequence_size = self.Sequence_size.value()
        FF_sequence_size = self.box_No_of_FFs.value()
        angular_range = self.angular_range.value()

        sample_pos = self.tomo_pos.value()
        FF_pos = self.FF_pos.value()
        #print('zero checked? ', self.zero_deg_checkBox.checkState())

        side_step_max = self.box_lateral_shift.value()                   # in µm

        #print(number_of_sequences, sequence_size, FF_sequence_size, angular_range, sample_pos, FF_pos, side_step_max)
        number_of_projections = number_of_sequences * sequence_size
        self.display_no_of_proj.display(number_of_projections)


        # COMPUTE LISTS #

        if (math.log(number_of_sequences, 2)) != int(math.log(number_of_sequences, 2)):
            #print('Choose from 2, 4, 8, 16, 32, 64, etc !!!')
            time.sleep(3)
            exit()
        grade = int(math.log(number_of_sequences, 2))


        # CREATE THETA-FIRST-LIST #

        theta_list = list(range(sequence_size * number_of_sequences))
        angular_large_steps = angular_range / sequence_size

        # 0!, 1/2, 1/4, 3/4, 1/8, 3/8, 5/8, 7/8, 1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16, 1/32, 3/32, ... , 31/33, 1/64, 3/64, etc...

        theta_first_list = list(range(2 ** grade))
        theta_first_list[0] = 0
        j = 1
        k = 1
        while (k < grade + 1):
            n = 1
            while (n < 2 ** k):
                theta_first_list[j] = n / (2 ** k)
                j = j + 1
                n = n + 2
            k = k + 1

        #print('theta_first_list = ',theta_first_list)


        #   POSITIONS   # ===================================================================================================== #   POSITIONS   #

        x_offset_list = list(range(sequence_size * number_of_sequences))

        k = 0
        while (k < number_of_sequences):  # Sequence loop

            l = 0
            while (l < sequence_size):
                x_offset_list[k * sequence_size + l] = side_step_max * 2 * (theta_first_list[k] - 0.5)
                l = l + 1
            k = k + 1


        #   ANGLES  # ========================================================================================================= #   ANGLES  #
        m = 0
        while (m < number_of_sequences):  # Sequence loop

            i = 0
            while (i < sequence_size):
                theta_list[m * sequence_size + i] = round(
                    ((theta_first_list[m] * angular_large_steps) + i * angular_large_steps), 4)
                i = i + 1

            m = m + 1
        #print(theta_list)


        #  PLOT  #==============================================================================================
        if (self.zero_deg_checkBox.checkState() == 2):
            plot_list = list(range(((sequence_size + FF_sequence_size+1) * number_of_sequences + FF_sequence_size) + 6))
            plot_list_pos = list(range(((sequence_size + FF_sequence_size+1) * number_of_sequences + FF_sequence_size) + 6))
        else:
            plot_list = list(range(((sequence_size + FF_sequence_size) * number_of_sequences + FF_sequence_size) + 6))
            plot_list_pos = list(range(((sequence_size + FF_sequence_size) * number_of_sequences + FF_sequence_size) + 6))

        #print(len(plot_list))
        self.display_no_of_images.display(len(plot_list))

        n = 3
        a = 1

        plot_list[0] = 0
        plot_list[1] = 90
        plot_list[2] = 180
        plot_list_pos[0] = sample_pos
        plot_list_pos[1] = sample_pos
        plot_list_pos[2] = sample_pos

        while (a < number_of_sequences * sequence_size + 1):
            if (a % sequence_size) == 1:
                j = 1  # FF counter
                while (j < FF_sequence_size + 1):
                    plot_list[n] = 0
                    plot_list_pos[n] = FF_pos
                    j = j + 1
                    n = n + 1

                if(self.zero_deg_checkBox.checkState() == 2):
                    plot_list[n] = 0
                    plot_list_pos[n] = sample_pos + x_offset_list[a - 1] / 1000
                    n = n + 1

            plot_list[n] = (theta_list[a - 1])
            plot_list_pos[n] = sample_pos + x_offset_list[a - 1] / 1000
            a = a + 1
            n = n + 1

        j = 1  # FF counter
        while (j < FF_sequence_size + 1):
            plot_list[n] = 0
            plot_list_pos[n] = FF_pos
            j = j + 1
            n = n + 1

        plot_list[n] = 0
        plot_list[n + 1] = 90
        plot_list[n + 2] = 180
        plot_list_pos[n] = sample_pos
        plot_list_pos[n + 1] = sample_pos
        plot_list_pos[n + 2] = sample_pos

        self.Graph.plot(list(range(len(plot_list))), plot_list, symbol='o', clear=True)
        self.Graph2.plot(list(range(len(plot_list_pos))), plot_list_pos, symbol='o', clear=True)

        return(theta_first_list, theta_list, x_offset_list, sample_pos, FF_pos, FF_sequence_size, number_of_sequences, sequence_size, plot_list_pos, plot_list)


    def generate_list_classic(self):

            # GET PARAMETERS #
            number_of_sequences_2 = self.No_of_sequences_2.value()
            sequence_size_2 = self.Sequence_size_2.value()
            FF_sequence_size_2 = self.box_No_of_FFs_2.value()

            sample_pos_2 = self.tomo_pos_2.value()
            FF_pos_2 = self.FF_pos_2.value()

            number_of_projections_2 = number_of_sequences_2 * sequence_size_2
            self.display_no_of_proj_2.display(number_of_projections_2)
            self.display_no_of_images_2.display((sequence_size_2 * number_of_sequences_2 + FF_sequence_size_2 * (number_of_sequences_2 + 1) + 6))


            #   NETTO-ANGLES  # ========================================================================================================= #   ANGLES  #

            angular_steps_2 = self.angular_range_2.value() / number_of_projections_2
            theta_list_2 = list(range(number_of_projections_2))
            m = 0
            while (m < number_of_projections_2):  # Sequence loop
                theta_list_2[m] = round(m * angular_steps_2, 4)
                m = m + 1
            #print(theta_list_2)


            #  PLOT  #==============================================================================================

            plot_list_2 = list(range(((sequence_size_2 + FF_sequence_size_2) * number_of_sequences_2 + FF_sequence_size_2) + 6))
            plot_list_pos_2 = list(range(((sequence_size_2 + FF_sequence_size_2) * number_of_sequences_2 + FF_sequence_size_2) + 6))

            #print(len(plot_list_2))
            self.display_no_of_images_2.display(len(plot_list_2))

            n = 3
            a = 1

            plot_list_2[0] = 0
            plot_list_2[1] = 90
            plot_list_2[2] = 180
            plot_list_pos_2[0] = sample_pos_2
            plot_list_pos_2[1] = sample_pos_2
            plot_list_pos_2[2] = sample_pos_2

            while (a < number_of_sequences_2 * sequence_size_2 + 1):
                if (a % sequence_size_2) == 1:
                    j = 1  # FF counter
                    while (j < FF_sequence_size_2 + 1):
                        plot_list_2[n] = (theta_list_2[a - 1])
                        plot_list_pos_2[n] = FF_pos_2
                        j = j + 1
                        n = n + 1

                plot_list_2[n] = (theta_list_2[a - 1])
                plot_list_pos_2[n] = sample_pos_2
                a = a + 1
                n = n + 1

            j = 1  # FF counter FFs at the end
            while (j < FF_sequence_size_2 + 1):
                plot_list_2[n] = 0
                plot_list_pos_2[n] = FF_pos_2
                j = j + 1
                n = n + 1

            plot_list_2[n] = 0
            plot_list_2[n + 1] = 90
            plot_list_2[n + 2] = 180
            plot_list_pos_2[n] = sample_pos_2
            plot_list_pos_2[n + 1] = sample_pos_2
            plot_list_pos_2[n + 2] = sample_pos_2

            self.Graph_2.plot(list(range(len(plot_list_2))), plot_list_2, symbol='o', clear=True)
            self.Graph2_2.plot(list(range(len(plot_list_pos_2))), plot_list_pos_2, symbol='o', clear=True)

            return(theta_list_2, sample_pos_2, FF_pos_2, FF_sequence_size_2, number_of_sequences_2, sequence_size_2, plot_list_pos_2, plot_list_2)


    def generate_list_refraction(self):

        # GET PARAMETERS #
        number_of_sequences_3 = int((self.No_of_sequences_3.currentText()))
        sequence_size_3 = self.Sequence_size_3.value()
        FF_sequence_size_3 = self.box_No_of_FFs_3.value()
        angular_range_3 = self.angular_range_3.value()

        sample_pos_3 = self.tomo_pos_3.value()
        FF_pos_3 = self.FF_pos_3.value()
        #print('zero checked? ', self.zero_deg_checkBox_3.checkState())

        side_step_max_3 = self.box_lateral_shift_3.value()                   # in µm

        #print(number_of_sequences_3, sequence_size_3, FF_sequence_size_3, angular_range_3, sample_pos_3, FF_pos_3, side_step_max_3)
        number_of_projections_3 = number_of_sequences_3 * sequence_size_3
        self.display_no_of_proj_3.display(number_of_projections_3)


        # COMPUTE LISTS #

        if (math.log(number_of_sequences_3, 2)) != int(math.log(number_of_sequences_3, 2)):
            #print('Choose from 2, 4, 8, 16, 32, 64, etc !!!')
            time.sleep(3)
            exit()
        grade_3 = int(math.log(number_of_sequences_3, 2))


        # CREATE THETA-FIRST-LIST #

        theta_list_3 = list(range(sequence_size_3 * number_of_sequences_3))
        angular_large_steps_3 = angular_range_3 / sequence_size_3

        # 0!, 1/2, 1/4, 3/4, 1/8, 3/8, 5/8, 7/8, 1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16, 1/32, 3/32, ... , 31/33, 1/64, 3/64, etc...

        theta_first_list_3 = list(range(2 ** grade_3))
        theta_first_list_3[0] = 0
        j = 1
        k = 1
        while (k < grade_3 + 1):
            n = 1
            while (n < 2 ** k):
                theta_first_list_3[j] = n / (2 ** k)
                j = j + 1
                n = n + 2
            k = k + 1

        #print('theta_first_list_3 = ',theta_first_list_3)


        #   POSITIONS   # ===================================================================================================== #   POSITIONS   #

        x_offset_list_3 = list(range(sequence_size_3 * number_of_sequences_3))

        k = 0
        while (k < number_of_sequences_3):  # Sequence loop

            l = 0
            while (l < sequence_size_3):
                x_offset_list_3[k * sequence_size_3 + l] = side_step_max_3 * 2 * (theta_first_list_3[k] - 0.5)
                l = l + 1
            k = k + 1


        #   ANGLES  # ========================================================================================================= #   ANGLES  #
        m = 0
        while (m < number_of_sequences_3):  # Sequence loop

            i = 0
            while (i < sequence_size_3):
                theta_list_3[m * sequence_size_3 + i] = round(
                    ((theta_first_list_3[m] * angular_large_steps_3) + i * angular_large_steps_3), 4)
                i = i + 1

            m = m + 1
        #print(theta_list_3)


        #  PLOT  #==============================================================================================
        if (self.zero_deg_checkBox_3.checkState() == 2):
            plot_list_3 = list(range(((sequence_size_3 + FF_sequence_size_3+1) * number_of_sequences_3 + FF_sequence_size_3) + 6))
            plot_list_pos_3 = list(range(((sequence_size_3 + FF_sequence_size_3+1) * number_of_sequences_3 + FF_sequence_size_3) + 6))
        else:
            plot_list_3 = list(range(((sequence_size_3 + FF_sequence_size_3) * number_of_sequences_3 + FF_sequence_size_3) + 6))
            plot_list_pos_3 = list(range(((sequence_size_3 + FF_sequence_size_3) * number_of_sequences_3 + FF_sequence_size_3) + 6))

        #print(len(plot_list_3))
        self.display_no_of_images_3.display(len(plot_list_3))

        n = 3
        a = 1

        plot_list_3[0] = 0
        plot_list_3[1] = 90
        plot_list_3[2] = 180
        plot_list_pos_3[0] = sample_pos_3
        plot_list_pos_3[1] = sample_pos_3
        plot_list_pos_3[2] = sample_pos_3

        while (a < number_of_sequences_3 * sequence_size_3 + 1):
            if (a % sequence_size_3) == 1:
                j = 1  # FF counter
                while (j < FF_sequence_size_3 + 1):
                    plot_list_3[n] = 0
                    plot_list_pos_3[n] = FF_pos_3
                    j = j + 1
                    n = n + 1

                if(self.zero_deg_checkBox_3.checkState() == 2):
                    plot_list_3[n] = 0
                    plot_list_pos_3[n] = sample_pos_3 + x_offset_list_3[a - 1] / 1000
                    n = n + 1

            plot_list_3[n] = (theta_list_3[a - 1])
            plot_list_pos_3[n] = sample_pos_3 + x_offset_list_3[a - 1] / 1000
            a = a + 1
            n = n + 1

        j = 1  # FF counter
        while (j < FF_sequence_size_3 + 1):
            plot_list_3[n] = 0
            plot_list_pos_3[n] = FF_pos_3
            j = j + 1
            n = n + 1

        plot_list_3[n] = 0
        plot_list_3[n + 1] = 90
        plot_list_3[n + 2] = 180
        plot_list_pos_3[n] = sample_pos_3
        plot_list_pos_3[n + 1] = sample_pos_3
        plot_list_pos_3[n + 2] = sample_pos_3

        self.Graph3.plot(list(range(len(plot_list_3))), plot_list_3, symbol='o', clear=True)
        self.Graph3_2.plot(list(range(len(plot_list_pos_3))), plot_list_pos_3, symbol='o', clear=True)

        return(theta_first_list_3, theta_list_3, x_offset_list_3, sample_pos_3, FF_pos_3, FF_sequence_size_3, number_of_sequences_3, sequence_size_3, plot_list_pos_3, plot_list_3)



    def get_Tomo_pos(self):
        self.tomo_pos.setValue(caget('PEGAS:miocb0101005.RBV'))

    def get_Tomo_pos_2(self):
        self.tomo_pos_2.setValue(caget('PEGAS:miocb0101005.RBV'))

    def get_Tomo_pos_3(self):
        self.tomo_pos_3.setValue(caget('PEGAS:miocb0102001.RBV'))

    def get_FF_pos(self):
        self.FF_pos.setValue(caget('PEGAS:miocb0101005.RBV'))

    def get_FF_pos_2(self):
        self.FF_pos_2.setValue(caget('PEGAS:miocb0101005.RBV'))

    def get_FF_pos_3(self):
        self.FF_pos_3.setValue(caget('PEGAS:miocb0102001.RBV'))



    #def window(self):
        app = QApplication(sys.argv)
        win = QWidget()
        button1 = QPushButton(win)
        button1.setText("Show dialog!")
        button1.move(50, 50)
        button1.clicked.connect(self.showDialog)
        win.setWindowTitle("Click button")
        win.show()
        sys.exit(app.exec_())

    #def showDialog(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText("Message box pop up window")
        msgBox.setWindowTitle("QMessageBox Example")
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msgBox.buttonClicked.connect(self.msgButtonClick)

        returnValue = msgBox.exec()
        if returnValue == QMessageBox.Ok:
            #print('OK clicked')

    #def msgButtonClick(i):
        #print("Button clicked is:", i.text())


    def save_files(self):

        theta_first_list, theta_list, x_offset_list, sample_pos, FF_pos, FF_sequence_size, number_of_sequences, sequence_size, plot_list_pos, plot_list = self.generate_list()
        zero_deg_proj = self.zero_deg_checkBox.checkState()

        path_out = QtWidgets.QFileDialog.getExistingDirectory()
        #print(path_out)
        file_name_theta = path_out + '/theta_list.txt'
        file_name_theta_first = path_out + '/theta_first_list.txt'
        file_name_X_offset = path_out + '/X_offset_list.txt'
        file_name_csv_list = path_out + '/csv_list.csv'
        file_name_parameter = path_out + '/parameter.csv'

        if len(os.listdir(path_out)) == 0:
            #print("Directory is empty")
        else:
            print("Directory is not empty!!!")

            #self.window()


        # CREATE AND SAVE CSV # =============================================================================================== # CREATE AND SAVE CSV #

        with open(file_name_csv_list, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar=' ')  # , quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['CT_MICOS_W', 'CT_MICOS_X'])
            n = 0
            while (n < len(plot_list_pos)):
                csv_writer.writerow(["{:.4f}".format(plot_list[n]),
                                     "{:.4f}".format(plot_list_pos[n])])
                n = n + 1


        # SAVE LISTS TO DISK # ================================================================================================ # SAVE LISTS TO DISK #

        np.savetxt(file_name_theta_first, theta_first_list)
        np.savetxt(file_name_X_offset, x_offset_list)
        np.savetxt(file_name_theta, theta_list)
        print('saving done')


        # SAVE PARAMETERS TO FILE # ==================================================================================== # SAVE PARAMETERS TO FILE #

        with open(file_name_parameter, mode = 'w', newline='') as parameter_file:
            csv_writer = csv.writer(parameter_file, delimiter = ' ', quotechar=' ')
            csv_writer.writerow(['number_of_sequences ', str(number_of_sequences),' '])
            csv_writer.writerow(['sequence_size ', str(sequence_size),' '])
            csv_writer.writerow(['FF_sequence_size ', str(FF_sequence_size),' '])
            csv_writer.writerow(['zero_deg_proj ', str(zero_deg_proj),' '])
            csv_writer.writerow(['angular_range ', str(self.angular_range.value()), ' '])
            csv_writer.writerow(['box_lateral_shift ', str(self.box_lateral_shift.value()), ' '])


    def save_files_classic(self):

        theta_list_2, sample_pos_2, FF_pos_2, FF_sequence_size_2, number_of_sequences_2, sequence_size_2, plot_list_pos_2, plot_list_2 = self.generate_list_classic()
        path_out_2 = QtWidgets.QFileDialog.getExistingDirectory()
        #print(path_out_2)
        file_name_theta_2 = path_out_2 + '/theta_list.txt'
        file_name_csv_list_2 = path_out_2 + '/csv_list.csv'
        file_name_parameter_2 = path_out_2 + '/parameter.csv'


        # CREATE AND SAVE CSV # =============================================================================================== # CREATE AND SAVE CSV #

        with open(file_name_csv_list_2, mode='w', newline='') as csv_file_2:
            csv_writer_2 = csv.writer(csv_file_2, delimiter=';', quotechar=' ')  # , quoting=csv.QUOTE_MINIMAL)
            csv_writer_2.writerow(['CT_MICOS_W', 'CT_MICOS_X'])
            n = 0
            while (n < len(plot_list_pos_2)):
                csv_writer_2.writerow(["{:.4f}".format(plot_list_2[n]),
                                     "{:.4f}".format(plot_list_pos_2[n])])
                n = n + 1


        # SAVE LIST TO DISK # ================================================================================================ # SAVE LISTS TO DISK #

        np.savetxt(file_name_theta_2, theta_list_2)
        print('saving done')


        # SAVE PARAMETERS TO FILE # ==================================================================================== # SAVE PARAMETERS TO FILE #

        with open(file_name_parameter_2, mode = 'w', newline='') as parameter_file_2:
            csv_writer_2 = csv.writer(parameter_file_2, delimiter = ' ', quotechar=' ')
            csv_writer_2.writerow(['number_of_sequences_2 ', str(number_of_sequences_2),' '])
            csv_writer_2.writerow(['sequence_size_2 ', str(sequence_size_2),' '])
            csv_writer_2.writerow(['angular_range_2 ', str(self.angular_range_2.value()),' '])
            csv_writer_2.writerow(['FF_sequence_size_2 ', str(FF_sequence_size_2),' '])


    def save_files_refraction(self):

        theta_first_list_3, theta_list_3, x_offset_list_3, sample_pos_3, FF_pos_3, FF_sequence_size_3, number_of_sequences_3, sequence_size_3, plot_list_pos_3, plot_list_3 = self.generate_list_refraction()
        zero_deg_proj_3 = self.zero_deg_checkBox_3.checkState()

        path_out_3 = QtWidgets.QFileDialog.getExistingDirectory()
        #print(path_out_3)
        file_name_theta_3 = path_out_3 + '/theta_list.txt'
        file_name_theta_first_3 = path_out_3 + '/theta_first_list.txt'
        file_name_X_offset_3 = path_out_3 + '/X_offset_list.txt'
        file_name_csv_list_3 = path_out_3 + '/csv_list_'
        file_name_parameter_3 = path_out_3 + '/parameter.csv'


        # CREATE AND SAVE CSV # =============================================================================================== # CREATE AND SAVE CSV #

        # creates for each sequence one csv-file
        n = 0
        for x in range(1, number_of_sequences_3 + 1):

            with open(file_name_csv_list_3 + str(x).zfill(2) + '.csv', mode='w', newline='') as csv_file_3:
                csv_writer_3 = csv.writer(csv_file_3, delimiter=';', quotechar=' ')  # , quoting=csv.QUOTE_MINIMAL)
                csv_writer_3.writerow(['6G_SAMPLE_ROLL', 'TOPO_MICOS_X'])

                i = 0

                if x == 1:
                    while (i < sequence_size_3 + FF_sequence_size_3 + 3 + zero_deg_proj_3/2):
                        csv_writer_3.writerow(["{:.4f}".format(plot_list_3[n]),
                                             "{:.4f}".format(plot_list_pos_3[n])])
                        n = n + 1
                        i = i + 1

                elif x == number_of_sequences_3:
                    while (i < sequence_size_3 + 2*FF_sequence_size_3 + 3 + zero_deg_proj_3/2):
                        csv_writer_3.writerow(["{:.4f}".format(plot_list_3[n]),
                                             "{:.4f}".format(plot_list_pos_3[n])])
                        n = n + 1
                        i = i + 1

                else:
                    while (i < sequence_size_3 + FF_sequence_size_3 + zero_deg_proj_3/2):
                        csv_writer_3.writerow(["{:.4f}".format(plot_list_3[n]),
                                             "{:.4f}".format(plot_list_pos_3[n])])
                        n = n + 1
                        i = i + 1


        # SAVE LISTS TO DISK # ================================================================================================ # SAVE LISTS TO DISK #

        np.savetxt(file_name_theta_first_3, theta_first_list_3)
        np.savetxt(file_name_X_offset_3, x_offset_list_3)
        np.savetxt(file_name_theta_3, theta_list_3)
        print('saving done')


        # SAVE PARAMETERS TO FILE # ==================================================================================== # SAVE PARAMETERS TO FILE #

        with open(file_name_parameter_3, mode = 'w', newline='') as parameter_file_3:
            csv_writer_3 = csv.writer(parameter_file_3, delimiter = ' ', quotechar=' ')
            csv_writer_3.writerow(['number_of_sequences ', str(number_of_sequences_3),' '])
            csv_writer_3.writerow(['sequence_size ', str(sequence_size_3),' '])
            csv_writer_3.writerow(['FF_sequence_size ', str(FF_sequence_size_3),' '])
            csv_writer_3.writerow(['zero_deg_proj ', str(zero_deg_proj_3),' '])
            csv_writer_3.writerow(['angular_range ', str(self.angular_range_3.value()), ' '])
            csv_writer_3.writerow(['box_lateral_shift ', str(self.box_lateral_shift_3.value()), ' '])





if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    main = CT_helper()
    main.show()
    sys.exit(app.exec_())
