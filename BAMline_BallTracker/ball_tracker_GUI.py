#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Use 'Agg' to generate PNGs of graphs without displaying via X.
# If X works, you can comment the following two lines:
#import matplotlib
#matplotlib.use('Agg')

from PyQt5 import QtCore, QtGui, QtWidgets
from balltracker.image import ImageStack
from balltracker.ball import ballSequence
from ball_tracker_questions import Ball_Tracker_Question

import tkinter.filedialog

def main():
	# Basics:
	# ---------------------------------------------------
	path_klick = tkinter.filedialog.askopenfilename(title="Select one file of the scan")

	name = "Kugeljustage_"

	import sys
	app = QtWidgets.QApplication(sys.argv)
	main_Ball_Tracker_Question = Ball_Tracker_Question()
	main_Ball_Tracker_Question.show()
	print('app.exec_()')
	app.exec_()

	print('Reading parameters...')
	startNumberProj = main_Ball_Tracker_Question.startNumberProj
	numberProj = main_Ball_Tracker_Question.numberProj
	startNumberFFs = main_Ball_Tracker_Question.startNumberFFs
	numberFFs = main_Ball_Tracker_Question.numberFFs
	Threshold = main_Ball_Tracker_Question.Threshold
	Binning = main_Ball_Tracker_Question.Binning     # or 2, whatever
	skip = main_Ball_Tracker_Question.skip
	print(Binning)

	htap = path_klick[::-1]
	path_in = path_klick[0: len(htap) - htap.find('/') - 1: 1]
	namepart = path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 5: 1]
	#counter = path_klick[len(htap) - htap.find('.') - 5: len(htap) - htap.find('.') - 1:1]
	#filetype = path_klick[len(htap) - htap.find('.') - 1: len(htap):1]

	inputFiles = ImageStack(filePattern= path_in + namepart + "%4d.tif", startNumber = startNumberProj, slices = numberProj)
	flatFields = ImageStack(filePattern= path_in + namepart + "%4d.tif", startNumber = startNumberFFs, slices = numberFFs)
	#darkFields = ImageStack(filePattern="BAMline_eveCSS/image_%4d.tif")
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

	seq.autoCrop(doAutoCrop=True, autoCropSize=900, autoCropBinningFactor=40)

	# Cropping the ball afterwards, mostly to produce an animation:
	seq.cropAndSaveCenteredBall(doCropBall=True, radius=300)


	# Drift compensation:
	# ---------------------------------------------------
	# To shift reference picture before applying it to a projection.
	# This needs a lot of RAM for full size images. Good idea to specify a drift observation ROI.

	seq.driftCompensation(refsToImg=False)

	# ROI region for unbinned image:
	# Define distance from [top or bottom] and [left or right], and a size.
	seq.driftROI(bottom=300, left=300, width=4000, height=600)


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