#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Use 'Agg' to generate PNGs of graphs without displaying via X.
# If X works, you can comment the following two lines:
#import matplotlib
#matplotlib.use('Agg')

from PyQt5 import QtCore, QtGui, QtWidgets
from balltracker.image import ImageStack
from balltracker.ball import ballSequence
from ball_tracker_questions_hdf import Ball_Tracker_Question
import tkinter.filedialog
import h5py
import numpy
import os
from PIL import Image


def main():
	# Basics:
	# ---------------------------------------------------
	path_klick = tkinter.filedialog.askopenfilename(title="Select one file of the scan", initialdir = "/mnt/raid/CT/")
	#path_klick = QtWidgets.QFileDialog.getOpenFileName(None, 'Select one file of the scan, please.',"/mnt/raid/CT/2021/2021_12/Markoetter/W500um/211207_1441_36_W500um___Z225_Y8400_20000eV_3p61um_100ms/")

	name = "Kugeljustage_"

	import sys
	app = QtWidgets.QApplication(sys.argv)
	main_Ball_Tracker_Question = Ball_Tracker_Question()
	main_Ball_Tracker_Question.show()
	print('app.exec_()')
	app.exec_()

	print('Reading parameters...')
	print(main_Ball_Tracker_Question.tabWidget)

	# checking start of rotation
	with h5py.File(path_klick, 'r') as hdf:
		entry = hdf.get('entry')
		w_data = entry.get('/entry/instrument/NDAttributes/CT_MICOS_W')
		w_array = numpy.array(w_data)
		print(w_array)
		i = main_Ball_Tracker_Question.HDF_FF1
		while i < len(w_array):
			print(i)
			if round(w_array[i]) != 0:  										# notice the last projection at zero degree
				last_zero_proj = i + 1 - main_Ball_Tracker_Question.HDF_FF1 	# start analysis 3 images after start of rotation
				break
			i = i + 1

	print('last zero projection: ', last_zero_proj)

	if main_Ball_Tracker_Question.tabWidget==0:
		startNumberProj = main_Ball_Tracker_Question.startNumberProj
		numberProj = main_Ball_Tracker_Question.numberProj
		startNumberFFs = main_Ball_Tracker_Question.startNumberFFs
		numberFFs = main_Ball_Tracker_Question.numberFFs

		Threshold = main_Ball_Tracker_Question.Threshold
		Binning = main_Ball_Tracker_Question.Binning     # or 2, whatever
		skip = main_Ball_Tracker_Question.skip
		print('Binning: ', Binning)

		htap = path_klick[::-1]
		path_in = path_klick[0: len(htap) - htap.find('/') - 1: 1]
		namepart = path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 5: 1]

		inputFiles = ImageStack(filePattern= path_in + namepart + "%4d.tif", startNumber = startNumberProj, slices = numberProj)
		flatFields = ImageStack(filePattern= path_in + namepart + "%4d.tif", startNumber = startNumberFFs, slices = numberFFs)
		darkFields = None
	else:
		hdf_FF1 = main_Ball_Tracker_Question.HDF_FF1
		hdf_Proj = main_Ball_Tracker_Question.HDF_Proj
		hdf_FF2 = main_Ball_Tracker_Question.HDF_FF2
		startNumberProj = hdf_FF1
		numberProj = hdf_Proj
		startNumberFFs = 0
		numberFFs = hdf_FF1

		Threshold = main_Ball_Tracker_Question.Threshold
		Binning = main_Ball_Tracker_Question.Binning     # or 2, whatever
		skip = main_Ball_Tracker_Question.skip
		print('Binning: ', Binning)

		htap = path_klick[::-1]
		path_in = path_klick[0: len(htap) - htap.find('/') - 1: 1]
		namepart = path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 5: 1]

		# creating temporal tiffs out of the hdf
		with h5py.File(path_klick, 'r') as hdf:
			entry = hdf.get('entry')
			data = entry.get('/entry/data/data')
			all_images = numpy.array(data)
			projections = all_images[startNumberProj:,:,:]
			FFs = all_images[0:hdf_FF1,:,:]

		# saving temporal projections
		i = 0
		folder = path_in + namepart + '_temp_proj/'
		if not os.path.exists(folder):
			os.makedirs(folder)

		while i < projections.shape[0]:
			filename_proj = path_in + namepart + '_temp_proj/Proj' + str(i).zfill(4) + '.tif'
			image_save = projections[i, :, :]
			img = Image.fromarray(image_save)
			img.save(filename_proj)
			i = i + 1

		# saving temporal FFs
		i = 0
		folder = path_in + namepart + '_temp_FFs/'
		if not os.path.exists(folder):
			os.makedirs(folder)
		while i < FFs.shape[0]:
			filename_FFs = path_in + namepart + '_temp_FFs/FFs' + str(i).zfill(4) + '.tif'
			image_save = FFs[i, :, :]
			img = Image.fromarray(image_save)
			img.save(filename_FFs)
			i = i + 1

		# setting paths for ball tracking
		inputFiles = ImageStack(filePattern= path_in + namepart + '_temp_proj/Proj' + "%4d.tif", startNumber = last_zero_proj, slices = numberProj)
		flatFields = ImageStack(filePattern= path_in + namepart + '_temp_FFs/FFs' + "%4d.tif", startNumber = 0, slices = numberFFs)
		darkFields = None

	outputFolder = path_in + "_Ball-Tracking"

	infosInFilename = True

	seq = ballSequence(inputFileStack=inputFiles, outputFolder=outputFolder, darkFileStack=darkFields, flatFileStack=flatFields)
	seq.setScanAngle(360)       # Used as fallback when no valid HDF file can be read.
	seq.skip(skip)                 # Only look at every n-th image. 1=all, 2=every second, ...

	# Display Mode: live image.
	# Show 'absorption', 'threshold' or 'edges' image. Or 'none'.
	# Works only when multiprocessing is turned off.
	seq.displayMode(mode='absorption')

	seq.saveIntermediates(True)  # Save intermediate processing steps as pictures.
	seq.showDebugInfo(True)     # Prints additional debug info to terminal.


	# Orientation:
	# ---------------------------------------------------
	# The 'orientation' tag of the tiff files is obeyed.
	# Further corrections can be made here if something is wrong:

	#seq.rotate("0")  # Rotate images by "90", "180" or "270" degress.
	#seq.flip(horizontal=False, vertical=False)


	# Multiprocessing:
	# ---------------------------------------------------
	# Python will automatically decide on the number of processes unless you set it.

	seq.multiprocessing(False)
	seq.numberOfProcesses(n=3)


	# Image preprocessing:
	# ---------------------------------------------------

	seq.applyDarks(False)  # Dark Field Correction
	seq.applyRefs(True)   # Flat Field Correction
	seq.median(3)
	seq.threshold(ratio=Threshold, absolute=None) # At least one of these parameters must be 'None'
	seq.patchCleanup(doCleanUp=True,
		             min_circle_area=100,
		             max_patch_area=None,
		             aspect_ratio_tolerance=0.15)
	seq.edgeDetection('sobel')   # 'sobel' (faster, thicker line) or 'canny' (slower, thinner line)


	# Data reduction:
	# ---------------------------------------------------

	seq.binning(Binning)
	seq.cropBorder(top=main_Ball_Tracker_Question.CropTop, bottom=main_Ball_Tracker_Question.CropBottom, left=main_Ball_Tracker_Question.CropLeft, right=main_Ball_Tracker_Question.CropRight)
	#seq.crop(x0=100, y0=1000, x1=5000, y1=2000)    # Crop overrides border crop, if defined.

	seq.autoCrop(doAutoCrop=True, autoCropSize=600, autoCropBinningFactor=40)

	# Cropping the ball afterwards, mostly to produce an animation:
	seq.cropAndSaveCenteredBall(doCropBall=True, radius=500)


	# Drift compensation:
	# ---------------------------------------------------
	# To shift reference picture before applying it to a projection.
	# This needs a lot of RAM for full size images. Good idea to specify a drift observation ROI.

	seq.driftCompensation(refsToImg=False)

	# ROI region for unbinned image:
	# Define distance from [top or bottom] and [left or right], and a size.
	seq.driftROI(bottom=300, left=300, width=4000, height=300)


	# ---------------------------------------------------
	# Maximum allowed deviation from circle fit,
	# in pixels, for unbinned image.

	seq.circleFitTolerances(max_meanDeviation=6, max_maxDeviation=30)


	# Intensity profile fit:
	# ---------------------------------------------------
	# Fits Beer-Lambert law to intensity profile, after circle fit.
	# This fit always uses the unbinned picture.
	# Fast; don't worry about computation time.

	seq.fitIntensityProfile(False)


	# Run the ball tracker to gather coordinates:
	# ---------------------------------------------------
	seq.trackBall()
	seq.saveParameters(name=name, infosInFilename=infosInFilename)
	seq.saveCoordinates(name=name, infosInFilename=infosInFilename)

	# Or import previously saved coordinate files.
	# The specifier tells which fit type shall be imported (or both).
	# Alternatively, use a single 'filename=' parameter to specify an
	# absolute path to a coordinate file.

	#seq.importCoordinates(name=name, specifier="coordinates_circleFit", infosInFilename=infosInFilename)
	#seq.importCoordinates(name=name, specifier="coordinates_intensityFit", infosInFilename=infosInFilename)


	# Calculate center of rotation + axis tilt from trajectory:
	# -----------------------------------------------------------
	seq.calcAxisParameters(fitSecondOrderWave=False)

	seq.saveCoordinates(name=name, infosInFilename=infosInFilename, withHeader=True)
	seq.saveGeometryResults(name=name, infosInFilename=infosInFilename)

	print("Finished in {}.".format(seq.getFormattedRuntime()))

	seq.plotTrajectories(displayPlot=True, savePlot=True, saveAs=name, infosInFilename=infosInFilename)


if __name__ == '__main__':
    main()