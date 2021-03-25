# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
print('starting imports')

from find_COR import COR_finder
from find_pixel_size import PixelSize_finder
from ask_once import ask_once
import tkinter.filedialog
import os


standard_path = "C:/Preview-CT_Messung/"

if __name__ == "__main__":

    import sys
    app = QtWidgets.QApplication(sys.argv)

    #path_klick = tkinter.filedialog.askopenfilename(title="Select one file of the scan", initialdir = standard_path)
    path_klick_dialog = QtWidgets.QFileDialog.getOpenFileName()
    path_klick = path_klick_dialog[0]
    print(path_klick)

    htap = path_klick[::-1]
    path_in = path_klick[0: len(htap) - htap.find('/') - 1: 1]
    ni_htap = path_in[::-1]
    last_path = path_klick[len(htap) - ni_htap.find('/') +len(ni_htap) - len(htap) - 1: len(htap) - htap.find('/') - 1: 1]
    print('last_path', last_path)
    namepart = path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 5: 1]
    file_name_parameter = path_in + '/parameter.csv'
    print(file_name_parameter)


    f = open(file_name_parameter, 'r')  # Reading scan-scheme parameters
    for line in f:
        line = line.strip()
        columns = line.split()
        print(columns[0])

        if str(columns[0]) == 'box_lateral_shift':
            box_lateral_shift = int(columns[1])
            print(columns[1])

        if str(columns[0]) == 'FF_sequence_size':
            FF_sequence_size = int(columns[1])
            print(columns[1])

    f.close()

    print('FF_sequence_size', FF_sequence_size)

    # ================================================================================================================ #
    print('prepare for showing ask_once')

    module_choice = ['interlaced_CT_preview_reco_movement_correction', 'interlaced_CT_preview_reco', 'interlaced_CT_normalization_only']#, 'interlaced_CT_32_bit_float_test']

    main_ask_once = ask_once(module_choice)
    main_ask_once.show()
    app.exec_()

    checkBox_save_normalized = main_ask_once.checkBox_save_normalized
    checkBox_classic_order = main_ask_once.checkBox_classic_order

    no_of_cores = main_ask_once.no_of_cores
    dark_field_value = main_ask_once.dark_field_value
    block_size = main_ask_once.block_size

    if main_ask_once.module_text == 'interlaced_CT_preview_reco':
        from interlaced_CT_16_bit_integer_test import CT_preview
    if main_ask_once.module_text == 'interlaced_CT_preview_reco_movement_correction':
        from interlaced_CT_16_bit_integer_movement_correction import CT_preview
    if main_ask_once.module_text == 'interlaced_CT_normalization_only':
        from interlaced_CT_normalization_only import CT_preview

    path_out = main_ask_once.path_out + last_path + '_evaluation'
    if os.path.isdir(path_out) is False:
        os.mkdir(path_out)
        print('make dir: ', path_out)

    index_COR_1 = 1
    index_COR_2 = 3
    FF_index = 4
    index_pixel_size_1 = 1
    index_pixel_size_2 = 4 + FF_sequence_size
    COR = main_ask_once.doubleSpinBox_COR
    print('COR', COR)
    rotate = main_ask_once.doubleSpinBox_Tilt
    pixel_size = main_ask_once.doubleSpinBox_PixelSize
    notknowCOR = main_ask_once.checkBox_notknowCOR
    notknowPixelSize = main_ask_once.checkBox_notknowPixelSize
    transpose = main_ask_once.transpose
    find_pixel_size_vertical = main_ask_once.find_pixel_size_vertical

    print('notknowCOR', notknowCOR)
    print('notknowPixelSize', notknowPixelSize)
    print('checkBox_save_normalized',checkBox_save_normalized)




    # ================================================================================================================ #
    if notknowCOR == True:
        print('prepare for showing COR_finder')
        main_COR_finder = COR_finder(path_klick, path_out, index_COR_1, index_COR_2, FF_index, transpose)
        value = main_COR_finder.show()
        print(value)
        app.exec_()
        COR = main_COR_finder.COR
        rotate = main_COR_finder.rotate
        print(COR)

    # ================================================================================================================ #
    if notknowPixelSize == True:

        if box_lateral_shift != 0:
            print('prepare for showing pixel_size_finder')
            main_find_pixel_size = PixelSize_finder(path_klick, path_out, index_pixel_size_1, index_pixel_size_2, FF_index, transpose, find_pixel_size_vertical)
            main_find_pixel_size.show()
            app.exec_()
            pixel_size = main_find_pixel_size.pixel_size
            print(pixel_size)
        else:
            pixel_size = 1
            print('no lateral shift')

    # ================================================================================================================ #
    print('prepare for showing CT-Preview')
    main_preview_CT = CT_preview(COR, rotate, pixel_size, path_klick, path_out, block_size, dark_field_value, no_of_cores, checkBox_save_normalized, checkBox_classic_order, transpose, find_pixel_size_vertical)#, algorithm, filter)
    main_preview_CT.show()
    app.exec_()


    print('call_programs DONE!')
    sys.exit(0)
    print('DONE')
