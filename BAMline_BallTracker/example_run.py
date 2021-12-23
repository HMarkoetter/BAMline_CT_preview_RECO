#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Use 'Agg' to generate PNGs of graphs without displaying via X.
# If X works, you can comment the following two lines:
#import matplotlib
#matplotlib.use('Agg')

from balltracker.image import ImageStack
from balltracker.ball import ballSequence

def main():
	# Basics:
	# ---------------------------------------------------

	name = "003_ball"

	inputFiles = ImageStack(filePattern="example_projections/003_ball_%6d_img.tif")
	flatFields = ImageStack(filePattern="example_projections/003_ball_%6d_ref.tif")
	darkFields = ImageStack(filePattern="example_projections/003_ball_%6d_dar.tif")

	outputFolder = "results"

	infosInFilename = True

	seq = ballSequence(inputFileStack=inputFiles, outputFolder=outputFolder, darkFileStack=darkFields, flatFileStack=flatFields)

	seq.setRowAxis("x")  # Name of detector's row vector axis
	seq.setColAxis("y")  # Name of detector's column vector axis / stage rotation axis
	seq.setBeamAxis("z") # Name of beam direction axis

	seq.setScanAngle(360)       # Rotation stage angular coverage for the given projections.
	seq.skip(2)                 # Only look at every n-th image. 1=all, 2=every second, ...

	# Display Mode: live image.
	# Show 'absorption', 'threshold' or 'edges' image. Or 'none'.
	# Works only when multiprocessing is turned off.
	seq.displayMode(mode='absorption') 

	seq.saveIntermediates(True)  # Save intermediate processing steps as pictures.
	seq.showDebugInfo(True)      # Prints additional debug info to terminal.


	# Orientation:
	# ---------------------------------------------------
	# The 'orientation' tag of the tiff files is obeyed.
	# Further corrections can be made here if something is wrong:

	seq.rotate("0")  # Rotate images by "90", "180" or "270" degress.
	seq.flip(horizontal=False, vertical=False)


	# Multiprocessing:
	# ---------------------------------------------------
	# Python will automatically decide on the number of processes unless you set it.

	seq.multiprocessing(True)
	seq.numberOfProcesses(n=3)


	# Image preprocessing:
	# ---------------------------------------------------

	seq.applyDarks(True)  # Dark Field Correction
	seq.applyRefs(True)   # Flat Field Correction
	seq.median(3)
	seq.threshold(ratio=0.7, absolute=None) # At least one of these parameters must be 'None'
	seq.patchCleanup(doCleanUp=True,
		             min_circle_area=(150*150),
		             max_patch_area=(1000*1000),
		             aspect_ratio_tolerance=0.15)
	seq.edgeDetection('sobel')   # 'sobel' (faster, thicker line) or 'canny' (slower, thinner line)


	# Data reduction:
	# ---------------------------------------------------

	seq.binning(2)
	seq.cropBorder(top=200, bottom=200, left=50, right=50)
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