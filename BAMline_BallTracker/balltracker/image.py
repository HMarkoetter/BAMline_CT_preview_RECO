# -*- coding: UTF-8 -*-

import numpy
import os    # File and path handling
import sys   # To get native byte order ('little' or 'big' endian?)
import math

# Scipy:
# 'ndimage' class for image processing
# 'optimize' class for intensity fit
# 'signal' class for drift analysis using FFT Convolution
from scipy import ndimage, optimize, stats, signal

# The 'feature' package from scikit-image,
# only needed for Canny edge detection, when used instead of Sobel.
from skimage.feature import canny   # Canny edge detection

from .general import *
from .tiffy import tiff

# Images are treated as 64-bit signed data types internally,
# to avoid out-of-range carries. Will be converted to desired
# data type when resulting image is written.
internalIntDataType   = numpy.dtype('int64')
internalFloatDataType = numpy.dtype('float64')

def isTIFF(filename):
    """Check if file name signifies a TIFF image."""
    if(filename.casefold().endswith('.tif') or filename.casefold().endswith('.tiff')):
        return True
    else:
        return False

def createImageStack(stack):
    """ Return an ImageStack object, if string is given. """
    if isinstance(stack, ImageStack):
        return stack
    elif isinstance(stack, str):
        return ImageStack(stack)
    elif stack == None:
        return None
    else:
        raise Exception("Not a valid image file stack definition: {}".format(stack))

class ImageFile:
    """Fundamental image file properties used for input and output."""

    def __init__(self, filename=None, dataType=None, byteOrder=None, flipByteOrder=False):
        self._filename  = None
        self._dataType  = None
        self._byteOrder = None   # 'little' or 'big' endian
        self._flipByteOrder = False

        self.setFilename(filename)
        self.setDataType(dataType)
        self.setByteOrder(byteOrder)
        self.setFlipByteOrder(flipByteOrder)

    def setFilename(self, filename):
        self._filename = filename

    def getFilename(self):
        return self._filename

    def getFileBasename(self):
        return os.path.basename(self._filename)

    def getDataType(self):
        return self._dataType

    def getByteOrder(self):
        return self._byteOrder

    def doFlipByteOrder(self):
        return self._flipByteOrder

    def setDataType(self, dataType):
        """ Set data type, either from numpy.dtype object or string. """
        if isinstance(dataType, numpy.dtype):
            self._dataType = dataType
        elif dataType == None:
            self._dataType = None
        elif isinstance(dataType, str):  # from string
            dt = numpy.dtype(dataType)
            self.setDataType(dt)
        else:
            raise Exception("{} is generally not a valid data type.".format(dataType))

    def setByteOrder(self, byteOrder):
        """ Set endianness, do sanity check before. """
        if byteOrder=='little' or byteOrder=='big' or byteOrder==None:
            self._byteOrder = byteOrder
        else:
            raise Exception("{} is not a valid byte order. Must be 'little' or 'big'.".format(byteOrder))

    def setFlipByteOrder(self, flipByteOrder):
        self._flipByteOrder = flipByteOrder

    def isInt(self):
        """ True if data type is supported int data type. """
        return numpy.issubdtype(self._dataType, numpy.integer)

    def isFloat(self):
        """ True if data type is supported float data type. """
        return numpy.issubdtype(self._dataType, numpy.floating)

class ImageROI:
    """ Defines a region of interest: upper left and lower right corner. """

    def __init__(self, x0, y0, x1, y1):
        self._x0 = 0
        self._y0 = 0
        self._x1 = 0
        self._y1 = 0
        self.set(x0, y0, x1, y1)

    def set(self, x0, y0, x1, y1):
        if x1 < x0:
            x0, x1 = x1, x0

        if y1 < y0:
            y0, y1 = y1, y0

        self._x0 = x0
        self._y0 = y0
        self._x1 = x1
        self._y1 = y1

    def x0(self):
        return self._x0

    def y0(self):
        return self._y0

    def x1(self):
        return self._x1

    def y1(self):
        return self._y1

    def width(self):
        return self._x1 - self._x0

    def height(self):
        return self._y1 - self._y0

    def area(self):
        return self.width()*self.height()

    def grow(self, amount):
        self.set(self._x0-amount, self._y0-amount, self._x1+amount, self._y1+amount)


class Image:
    """ Stores pixel data, provides image processing routines. """

    def __init__(self, inputFile=None, outputFile=None):
        self._inputFile  = None   # type ImageFile
        self._outputFile = None   # type ImageFile
        self._px         = 0  # 2D numpy array that contains the pixel values.
        self._height     = 0  # Image height in px.
        self._width      = 0  # Image width in px.
        self._index      = 0  # Slice number in a 3D volume.

        self._rotation     = None
        self._flipHorz     = False
        self._flipVert     = False

        self._n_accumulations = 0   # Counts number of accumulated pictures for averaging (mean)
        self._boundingBoxX0   = 0   # After cropping: bounding box offset relative to original image.
        self._boundingBoxY0   = 0
        self._resolution      = 1   # After binning: new resolution relative to original image.

        self.setInputFile(inputFile)
        self.setOutputFile(outputFile)

    def __del__(self):
        """ Delete pixel map upon object destruction. """
        self._px =0

    def setInputFile(self, inputFile):
        """ Set input file properties from ImageFile object or string. """
        if isinstance(inputFile, ImageFile) or (inputFile == None):
            self._inputFile = inputFile
        elif isinstance(inputFile, str):  # string given
            self._inputFile = ImageFile(inputFile)
        else:
            raise Exception("{} is not a valid file identifier.")

    def setOutputFile(self, outputFile):
        """ Set output file properties from ImageFile object or string. """
        if isinstance(outputFile, ImageFile) or (outputFile == None):
            self._outputFile = outputFile
        elif isinstance(outputFile, str):  # string given
            self._outputFile = ImageFile(outputFile)
        else:
            raise Exception("{} is not a valid file identifier.")

    def setHeight(self, height):
        """ Set image height in px. """
        self._height = height

    def setWidth(self, width):
        """ Set image width in px. """
        self._width = width

    def setIndex(self, index):
        """ Set image index position in 3D stack (in px). """
        self._index = index



    def shape(self, width, height, index, dataType=None, value=0):
        """ Re-format image to given dimensions and data type. """
        self.setWidth(width)
        self.setHeight(height)
        self.setIndex(index)

        if dataType == None:
            dataType = self.getInternalDataType()

        self._px = numpy.full((height, width), fill_value=value, dtype=dataType)

    def erase(self, value=0):
        """ Set all pixels to 'value'. """
        w = self.getWidth()
        h = self.getHeight()
        self._px = 0
        self._px = numpy.full((h, w), fill_value=value, dtype=self.getInternalDataType())
   
    def getPixelMap(self):
        return self._px

    def setPixelMap(self, px):
        self._px = px

    def setPixel(self, x, y, value):
        self._px[y][x] = value

    def getPixel(self, x, y):
        return self._px[y][x]

    def isSet(self):
        """ Check if image has a valid width and height. """
        if(self.getHeight() > 0):
            if(self.getWidth() > 0):
                return True

        return False

    def getWidth(self):
        return self._width

    def getHeight(self):
        return self._height

    def getNPixels(self):
        """ Calculate number of pixels in image. """
        return (self.getWidth() * self.getHeight())

    def getIndex(self):
        return self._index

    def getBoundingBoxX0(self):
        return self._boundingBoxX0

    def getBoundingBoxY0(self):
        return self._boundingBoxY0

    def getResolution(self):
        return self._resolution

    def getFileByteOrder(self):
        return self._fileByteOrder

    def getMax(self):
        """ Return maximum intensity in image. """
        return numpy.amax(self._px)

    def getMin(self):
        """ Return minimum intensity in image. """
        return numpy.amin(self._px)

    def getMean(self):
        """ Return arithmetic mean of the image grey values. """
        return numpy.mean(self._px)

    def getStdDev(self):
        """ Return the standard deviation of the image grey values. """
        return numpy.std(self._px)

    def setRotation(self, rotation):
        self._rotation = rotation

    def getRotation(self):
        return self._rotation

    def rot90(self):
        if self.isSet():
            self._px = numpy.rot90(self._px, k=1)
            self._width, self._height = self._height, self._width

    def rot180(self):
        if self.isSet():
            self._px = numpy.rot90(self._px, k=2)

    def rot270(self):
        if self.isSet():
            self._px = numpy.rot90(self._px, k=-1)
            self._width, self._height = self._height, self._width

    def rotate(self, rotation):
        if rotation == None:
            rotation = self._rotation
        else:
            self.setRotation(rotation)

        if rotation == "90":
            self.rot90()
        elif rotation == "180":
            self.rot180()
        elif rotation == "270":
            self.rot270()

    def flipHorizontal(self):
        self._flipHorz = not self._flipHorz
        if self.isSet():
            self._px = numpy.fliplr(self._px)

    def flipVertical(self):
        self._flipVert = not self._flipVert
        if self.isSet():
            self._px = numpy.flipud(self._px)

    def setFlip(self, horz=False, vert=False):
        self._flipHorz = horz
        self._flipVert = vert

    def getHorizontalFlip(self):
        return self._flipHorz

    def getVerticalFlip(self):
        return self._flipVert

    def flip(self, horizontal=False, vertical=False):
        if horizontal:
            self.flipHorizontal()
        if vertical:
            self.flipVertical()

    def getInternalDataType(self):
        """ Data type used internally for all image data. """
        return numpy.dtype('float64')

    def containsPixelValue(self, value):
        """ Check if image contains a certain grey value. """
        return numpy.any(self._px == value)

    def dimensionsMatch(self, img):
        """ Check if image dimensions match with another image. """
        if self.isSet() and img.isSet():
            if(self.getHeight() == img.getHeight()):
                if(self.getWidth() == img.getWidth()):
                    return True

        raise Exception("Pixel dimensions do not match: {}x{} vs. {}x{}".format(self.getWidth(), self.getHeight(), img.getWidth(), img.getHeight()))
        
        return False

    def read(self, filename=None):
        """ Read TIFF or RAW, decide by file name. """
        if filename == None:
            filename = self._inputFile.getFilename()
        else:
            self.setFilename(filename)

        # If no internal file name is specified, do nothing.
        if filename == None:
            return

        if isTIFF(self._inputFile.getFilename()):
            self.readTIFF(self._inputFile.doFlipByteOrder())
        else:
            self.readRAW(self.getWidth(), self.getHeight(), self.getIndex(), self._inputFile.getDataType(), self._inputFile.getByteOrder())

    def readTIFF(self, flipByteOrder=False, obeyOrientation=True):
        """ Import TIFF file. """
        if os.path.isfile(self._inputFile.getFilename()):
            basename = self._inputFile.getFileBasename()
            
            tiffimg = tiff()
            tiffimg.read(self._inputFile.getFilename())
            img = tiffimg.imageData(subfile=0, channel=0, obeyOrientation=obeyOrientation)  # get a greyscale image from TIFF subfile 0
            width = tiffimg.getWidth(subfile=0)
            height = tiffimg.getHeight(subfile=0)

            self._inputFile.setDataType(img.dtype) 

            if flipByteOrder:
                img.byteswap(inplace=True)

            # Convert to internal data type for either int or float:
            self._px = img.astype(self.getInternalDataType())

            # Check if array in memory has the dimensions stated in the TIFF file:
            if((height == len(self._px)) and (width == len(self._px[0]))):
                self.setHeight(height)
                self.setWidth(width)
            else:
                raise Exception("Width ({}px) and height ({}px) from the TIFF header do not match the data width ({}px) and height ({}px) that has been read.".format(width, height, len(self._px[0]), len(self._px)))
        else:
            raise Exception("Can't find " + self._inputFile.getFilename())

    def readRAW(self, width, height, index=0, dataType=None, byteOrder=None, fileHeaderSize=0, imageHeaderSize=0):
        """ Import RAW image file. """
        if not isinstance(self._inputFile, ImageFile):
            raise Exception("No valid input file defined.")

        if dataType == None:
            dataType = self._inputFile.getDataType()
        else:
            self._inputFile.setDataType(dataType)

        if byteOrder == None:
            byteOrder = self._inputFile.getByteOrder()
            if byteOrder == None:
                byteOrder = sys.byteorder

        self._inputFile.setByteOrder(byteOrder)

        if os.path.isfile(self._inputFile.getFilename()):
            self.shape(width, height, index, self._inputFile.getDataType())

            basename = self._inputFile.getFileBasename()
            #log("Reading RAW file {}...".format(basename))

            byteOffset = fileHeaderSize + (index+1)*imageHeaderSize + index*(self.getNPixels() * self._inputFile.getDataType().itemsize)

            with open(self._inputFile.getFilename(), 'rb') as f:
                f.seek(byteOffset)
                self._px = numpy.fromfile(f, dtype=self._inputFile.getDataType(), count=self.getNPixels(), sep="")

            # Treat endianness. If the native byte order of the system is different
            # than the given file byte order, the bytes are swapped in memory
            # so that it matches the native byte order.
            nativeEndian = sys.byteorder
            if nativeEndian == 'little':
                if byteOrder == 'big':
                    self._px.byteswap(inplace=True)
            elif nativeEndian == 'big':
                if byteOrder == 'little':
                    self._px.byteswap(inplace=True)

            # Convert to internal data type:
            self._px = self._px.astype(self.getInternalDataType())

            # Reshape to 2D array:
            self._px = numpy.reshape(self._px, (height, width))

        else:
            raise Exception("Can't find " + self._inputFile.getFilename())

    def getDataTypeClippingBoundaries(self, dataType):
        # Get clipping boundaries if grey values have to be
        # clipped to the interval supported by the int image type:
        clipMin = 0
        clipMax = 1
        if numpy.issubdtype(dataType, numpy.integer):
            intInfo   = numpy.iinfo(dataType)
            clipMin   = intInfo.min
            clipMax   = intInfo.max
        elif numpy.issubdtype(dataType, numpy.floating):
            floatInfo = numpy.finfo(dataType)
            clipMin   = floatInfo.min
            clipMax   = floatInfo.max

        return clipMin, clipMax

    def touchFolder(self, filename):
        """ Check if folder exists. Otherwise, create. """
        folder  = os.path.dirname(filename)
        if folder == "" or folder == None:
            folder = "."
        if not os.path.exists(folder):
            os.makedirs(folder)

    def save(self, filename=None, fileDataType=None, fileByteOrder=None, appendChunk=False, clipValues=True):
        """ Save image as TIFF or RAW. """
        if not isinstance(self._outputFile, ImageFile):
            self._outputFile = ImageFile()

        if (filename == None) or (filename == ""):
            filename = self._outputFile.getFilename()
            if (filename == None) or (filename == ""):
                raise Exception("No output file name specified.")
        else:
            self._outputFile.setFilename(filename)

        if fileDataType == None:
            fileDataType = self._outputFile.getDataType()
            if fileDataType == None:
                if isinstance(self._inputFile, ImageFile):
                    fileDataType = self._inputFile.getDataType()
                    if(fileDataType != None):
                        self._outputFile.setDataType(fileDataType)
                    else:
                        raise Exception("Please specify a data type for the output file: {filename}".format(filename=filename))
                else:
                    raise Exception("Please specify a data type for the output file: {filename}".format(filename=filename))
        else:
            self._outputFile.setDataType(fileDataType)

        if fileByteOrder == None:
            fileByteOrder = self._outputFile.getByteOrder()
            if fileByteOrder == None:
                if isinstance(self._inputFile, ImageFile):
                    fileByteOrder = self._inputFile.getByteOrder()
                    self._outputFile.setByteOrder(fileByteOrder)

            if fileByteOrder == None:
                fileByteOrder = "little"

        self._outputFile.setByteOrder(fileByteOrder)

        
        if isTIFF(filename):
            self.saveTIFF(filename, fileDataType, clipValues)
        else:
            self.saveRAW(filename, fileDataType, fileByteOrder, appendChunk, clipValues, addInfo=False)

    def saveTIFF(self, filename=None, fileDataType=None, clipValues=True):
        if (filename != None) and (len(filename) > 0):
            fileBaseName = os.path.basename(filename)
            if (fileBaseName == "") or (fileBaseName == None):
                raise Exception("No output file name specified for the image to be saved.")

            if fileDataType != None:
                if not isTIFF(filename):
                    filename += ".tif"

                self.touchFolder(filename)
                
                tiffdata = None
                if clipValues:  # Clipping
                    clipMin, clipMax = self.getDataTypeClippingBoundaries(fileDataType)
                    tiffdata = numpy.clip(self._px, clipMin, clipMax).astype(fileDataType)
                else:  # No clipping or float
                    tiffdata = self._px.astype(fileDataType)

                tiffimg = tiff()
                tiffimg.set(tiffdata)
                tiffimg.save(filename=filename, endian='little')
            else:
                raise Exception("Please specify a data type for the output file: {filename}".format(filename=filename))
        else:
            raise Exception("No output file name specified for the image to be saved.")
            
    def saveRAW(self, filename=None, fileDataType=None, fileByteOrder=None, appendChunk=False, clipValues=True, addInfo=False):
        if (filename != None) and (len(filename) > 0):
            fileBaseName = os.path.basename(filename)
            if (fileBaseName == "") or (fileBaseName == None):
                raise Exception("No output file name specified for the image to be saved.")

            if fileDataType != None:
                if fileByteOrder == None:
                    fileByteOrder = "little"

                # Reshape to 1D array and convert to file data type (from internal 64bit data type)
                outBytes = numpy.reshape(self._px, int(self._width)*int(self._height))

                if clipValues:  # Clipping
                    clipMin, clipMax = self.getDataTypeClippingBoundaries(fileDataType)
                    outBytes = numpy.clip(outBytes, clipMin, clipMax)

                outBytes = outBytes.astype(fileDataType)

                # Treat endianness. If the native byte order of the system is different
                # than the desired file byte order, the bytes are swapped in memory
                # before writing to disk.
                nativeEndian = sys.byteorder
                if nativeEndian == 'little':
                    if fileByteOrder  == 'big':
                        outBytes.byteswap(inplace=True)
                elif nativeEndian == 'big':
                    if fileByteOrder == 'little':
                        outBytes.byteswap(inplace=True)

                if addInfo:
                    shortEndian = "LE"
                    if fileByteOrder == "big":
                        shortEndian = "BE"

                    infoString = "_{width}x{height}_{dataType}_{endian}".format(width=self._width, height=self._height, dataType=fileDataType, endian=shortEndian)

                    basename, extension = os.path.splitext(filename)
                    filename = basename + infoString + extension

                self.touchFolder(filename)
                if not appendChunk:  # save as single raw file
                    with open(filename, 'w+b') as file:
                        file.write(outBytes)
                        file.close()
                    #outBytes.tofile(filename, sep="")
                else: # append to the bytes of the chunk file
                    with open(filename, 'a+b') as file:
                        file.write(outBytes)
                        file.close()
            else:
                raise Exception("Please specify a data type for the output file: {filename}".format(filename=filename))
        else:
            raise Exception("No output file name specified for the image to be saved.")

    def calcRelativeShift(self, referenceImage):
        if self.dimensionsMatch(referenceImage):
            # Convolution of this pixmap with the vertically and horizontally mirrored reference pixmap
            img1 = self._px - int(numpy.mean(self._px))
            img2 = referenceImage.getPixelMap() - numpy.mean(referenceImage.getPixelMap())

            convolution = signal.fftconvolve(img1, img2[::-1,::-1], mode='same')

            maximum = numpy.unravel_index(numpy.argmax(convolution), convolution.shape)

            return (maximum[1] - self.getWidth()/2, maximum[0] - self.getHeight()/2)
        else:
            raise Exception("Dimensions of image ({}, {}) and reference image ({}, {}) must match for convolution.".format(self.getWidth(), self.getHeight(), referenceImage.getWidth(), referenceImage.getHeight()))

    def getShiftedPixmap(self, xShift, yShift):
        return ndimage.interpolation.shift(self._px, (int(xShift), int(yShift)), mode='nearest')

    def accumulate(self, addImg, compensateShift=False, roiX0=None, roiY0=None, roiX1=None, roiY1=None):
        if (compensateShift == True) and (self._n_accumulations > 0):
            shift = (0, 0)

            if (roiX0 == None) or (roiY0 == None) or (roiX1 == None) or (roiY1 == None):
                shift = self.calcRelativeShift(addImg)
            else:
                # Crop image to drift ROI,
                croppedRef = copy.deepcopy(self)
                croppedRef.crop(x0=roiX0, y0=roiY0, x1=roiX1, y1=roiY1)

                croppedImg = copy.deepcopy(addImg)
                croppedImg.crop(x0=roiX0, y0=roiY0, x1=roiX1, y1=roiY1)

                shift = croppedImg.calcRelativeShift(croppedRef)

            log("Shift: {}".format(shift))
            shiftedPixMap = addImg.getShiftedPixmap(shift[1], shift[0])
            addImg.setPixelMap(shiftedPixMap)

        if self._n_accumulations == 0:
            self.setPixelMap(addImg.getPixelMap())
        else:
            if (self.dimensionsMatch(addImg)):
                self._px += addImg.getPixelMap()
            else:
                raise Exception("Current pixel dimensions ({currentX}x{currentY}) don't match dimensions of new file ({newX}x{newY}): {filename}".format(currentX=self.getWidth(), currentY=self.getHeight(), newX=addImg.getWidth(), newY=addImg.getHeight(), filename=addImg._inputFile.getFilename()))

        self._n_accumulations += 1

    def resetAccumulations(self):
        self._n_accumulations = 0

    def averageAccumulations(self):
        if self._n_accumulations > 1:
            self._px = self._px / self._n_accumulations
            log("Accumulated and averaged {} images.".format(self._n_accumulations))
            self._n_accumulations = 1

    def applyDark(self, dark):
        """ Apply dark image correction (offset). """
        if self.dimensionsMatch(dark):
            self._px = self._px - dark.getPixelMap()
        else:
            raise Exception("The dimensions of the image do not match the dimensions of the dark image for offset correction.")

    def applyFlatfield(self, ref, rescaleFactor):
        """ Apply flat field correction (free beam white image / gain correction). """
        if self.dimensionsMatch(ref):
            if(not ref.containsPixelValue(0)):  # avoid division by zero
                self._px = (self._px / ref.getPixelMap()) * float(rescaleFactor)
            else: # avoid division by zero
                self._px = (self._px / numpy.clip(ref.getPixelMap(), 0.1, None)) * float(rescaleFactor)
        else:
            raise Exception("The dimensions of the image do not match the dimensions of the flat image for flat field correction.")

    def horizontalProfile(self, yPos):
        if yPos < self.getHeight():
            return self._px[yPos]
        else:
            raise Exception("Requested position for horizontal profile is out of bounds: y={} in an image that has {} rows.".format(yPos, self.getHeight()))

    def horizontalROIProfile(self, ROI):
        # Take full image if no ROI is given
        if ROI==None:
            ROI = ImageROI(0, 0, self.getWidth(), self.getHeight())

        slc = self._px[ROI.y0():ROI.y1(), ROI.x0():ROI.x1()]

        profile = slc.mean(axis=0)
        return profile

    def clip(self, lower, upper):
        """ Clip grey values to given boundary interval. """
        self._px = numpy.clip(self._px, lower, upper)

    def crop(self, x0, y0, x1, y1):
        """ Crop to given box (x0, y0)--(x1, y1). """
        if x0 > x1:
            x0,x1 = x1,x0

        if y0 > y1:
            y0,y1 = y1,y0

        if y1 > self.getHeight()  or  x1 > self.getWidth():
            raise Exception("Trying to crop beyond image boundaries.")

        self._boundingBoxX0 += x0
        self._boundingBoxY0 += y0

        self._px = self._px[int(y0):int(y1),int(x0):int(x1)]   # Array has shape [y][x]
        self._width  = int(x1 - x0)
        self._height = int(y1 - y0)

    def cropBorder(self, top=0, bottom=0, left=0, right=0):
        """ Crop away given border around image. """
        x0 = int(left)
        y0 = int(top)
        x1 = self.getWidth() - int(right)
        y1 = self.getHeight() - int(bottom)

        self.crop(x0, y0, x1, y1)

    def cropROIaroundPoint(self, centerX, centerY, roiWidth, roiHeight):
        """ Crop a region of interest, centred around given point. """
        centerX = int(centerX)
        centerY = int(centerY)
        roiWidth = int(roiWidth)
        roiHeight= int(roiHeight)

        if roiWidth < 0:
            roiWidth = abs(roiWidth)
        if roiHeight < 0:
            roiHeight = abs(roiHeight)
        if roiWidth == 0 or roiHeight == 0:
            raise Exception("The region of interest should not be a square of size 0.")

        x0 = centerX - roiWidth/2
        x1 = centerX + roiWidth/2
        y0 = centerY - roiHeight/2
        y1 = centerY + roiHeight/2

        if x1<0 or y1<0:
            raise Exception("Right or lower boundary for ROI (x1 or y1) cannot be below zero.")

        if roiWidth>self.getWidth() or roiHeight>self.getHeight():
            raise Exception("Size of the ROI is bigger than the image size. ROI: " + str(roiWidth) + " x " + str(roiHeight) + ". Image: " + str(self.getWidth()) + " x " + str(self.getHeight()))   
        if x0 < 0:
            x1 += abs(x0)
            x0 = 0

        if y0 < 0:
            y1 += abs(y0)
            y0 = 0

        if x1 >= self.getWidth():
            x1 = self.getWidth()
            x0 = x1 - roiWidth

        if y1 >= self.getHeight():
            y1 = self.getHeight()
            y0 = y1 - roiHeight

        # These should match roiWidth and roiHeight...
        roiDimX = x1 - x0
        roiDimY = y1 - y0

        self.crop(x0, y0, x1, y1)
        return x0, x1, y0, y1

    def bin(self, binSizeX, binSizeY, operation="mean"):
        """ Decrease image size by merging pixels using specified operation.
            Valid operations: mean, max, min, sum. """

        if binSizeX == None:
            binSizeX = 1

        if binSizeY == None:
            binSizeY = 1

        if (binSizeX > 1) or (binSizeY > 1):
            # Picture dimensions must be integer multiple of binning factor. If not, crop:
            overhangX = math.fmod(int(self.getWidth()), binSizeX)
            overhangY = math.fmod(int(self.getHeight()), binSizeY)
            if (overhangX > 0) or (overhangY > 0):
                #log("Cropping before binning because of nonzero overhang: (" + str(overhangX) + ", " + str(overhangY) + ")")
                self.crop(0, 0, self.getWidth()-int(overhangX), self.getHeight()-int(overhangY))

            newWidth  = self._width // binSizeX
            newHeight = self._height // binSizeY

            # Shift pixel values that need to be binned together into additional axes:
            binshape = (newHeight, binSizeY, newWidth, binSizeX)
            self._px = self._px.reshape(binshape)
            
            # Perform binning operation along binning axes (axis #3 and #1).
            # These axes will be collapsed to contain only the result
            # of the binning operation.
            if operation == "mean":
                self._px = self._px.mean(axis=(3, 1))
            elif operation == "sum":
                self._px = self._px.sum(axis=(3, 1))
            elif operation == "max":
                self._px = self._px.max(axis=(3, 1))
            elif operation == "min":
                self._px = self._px.min(axis=(3, 1))
            elif operation == None:
                raise Exception("No binning operation specified.")
            else:
                raise Exception("Invalid binning operation: {}.".format(operation))

            self.setWidth(newWidth)
            self.setHeight(newHeight)

            # Resolution assumes isotropic pixels...
            self._resolution *= binSizeX

    def addImage(self, other):
        """ Add pixel values from another image to this image. """
        if self.dimensionsMatch(other):
            self._px = self._px + other.getPixelMap()

    def subtractImage(self, other):
        """ Add pixel values from another image to this image. """
        if self.dimensionsMatch(other):
            self._px = self._px - other.getPixelMap()

    def add(self, value):
        self._px += value

    def subtract(self, value):
        self._px -= value

    def multiply(self, value):
        self._px *= value

    def divide(self, value):
        """ Divide all pixels values by given scalar value. """
        self._px = self._px.astype(internalFloatDataType) / float(value)

    def invert(self, min=0, maximum=65535):
        self._px = maximum - self._px

    def max(self):
        return self._px.max()

    def min(self):
        return self._px.min()

    def renormalize(self, newMin=0, newMax=1, currentMin=None, currentMax=None, ROI=None):
        """Renormalization of grey values from (currentMin, Max) to (newMin, Max) """

        # Take full image if no ROI is given
        if ROI==None:
            ROI = ImageROI(0, 0, self.getWidth(), self.getHeight())

        slc = self._px[ROI.y0():ROI.y1(), ROI.x0():ROI.x1()]

        if currentMin == None:
            currentMin = slc.min()

        if currentMax == None:
            currentMax = slc.max()

        if(currentMax != currentMin):
            slc = (slc-currentMin)*(newMax-newMin)/(currentMax-currentMin)+newMin
            self._px[ROI.y0():ROI.y1(), ROI.x0():ROI.x1()] = slc
        else:
            slc = slc*0
            self._px[ROI.y0():ROI.y1(), ROI.x0():ROI.x1()] = slc
            #raise Exception("Division by zero upon renormalization: currentMax=currentMin={}".format(currentMax))


    def stats(self, ROI=None):
        """ Image or ROI statistics. Mean, Standard Deviation """

        # Take full image if no ROI is given
        if ROI==None:
            ROI = ImageROI(0, 0, self.getWidth(), self.getHeight())

        slc = self._px[ROI.y0():ROI.y1(), ROI.x0():ROI.x1()]

        mean  = numpy.mean(slc)
        sigma = numpy.std(slc)

        return {"mean": mean, "stddev": sigma, "width": ROI.width(), "height": ROI.height(), "area": ROI.area()}

    def applyMedian(self, kernelSize=1):
        if kernelSize > 1:
            self._px = ndimage.median_filter(self._px, int(kernelSize))

    def applyThreshold(self, threshold, lower=0, upper=65535):
        self._px = numpy.where(self._px > threshold, upper, lower)

    def renormalizeToMeanAndStdDev(self, mean, stdDev, ROI=None):
        """ Renormalize grey values such that mean=30000, (mean-stdDev)=0, (mean+stdDev)=60000 """

        # Take full image if no ROI is given
        if ROI==None:
            ROI = ImageROI(0, 0, self.getWidth(), self.getHeight())

        self._px[ROI.y0():ROI.y1(), ROI.x0():ROI.x1()] = ((self._px[ROI.y0():ROI.y1(), ROI.x0():ROI.x1()].astype(internalFloatDataType) - mean)/stdDev)*30000 + 30000

    def edges_sobel(self):
        # Sobel edge detection:
        edgesX = ndimage.sobel(self._px, axis=0, mode='nearest')
        edgesY = ndimage.sobel(self._px, axis=1, mode='nearest')
        return numpy.sqrt(edgesX**2 + edgesY**2)

    def edges_canny(self):
        # Canny edge detection. Needs 'scikit-image' package.  from skimage import feature
        return canny(self._px)

    def filter_edges(self, mode='sobel'):
        if(mode == 'sobel'):
            self._px = self.edges_sobel()
        elif(mode == 'canny'):
            self._px = self.edges_canny()
        else:
            raise Exception("Valid edge detection modes: 'sobel' or 'canny'")
        
        # Rescale:
        self._px = self._px.astype(self.getInternalDataType())
        #self.thresholding(0)    # black=0, white=65535

    def cleanPatches(self, min_patch_area=None, max_patch_area=None, remove_border_patches=False, aspect_ratio_tolerance=None):
        iterationStructure = ndimage.generate_binary_structure(rank=2, connectivity=2)  # apply to rank=2D array, only nearest neihbours (connectivity=1) or next nearest neighbours as well (connectivity=2)

        labelField, nPatches = ndimage.label(self._px, iterationStructure)
        nCleaned   = 0
        nRemaining = 0
        patchGeometry = []

        if nPatches == 0:
            log("Found no structures")
        else:
            self.erase()

            areaMin = 0
            if(min_patch_area != None):
                areaMin = min_patch_area
            
            areaMax = self.getWidth() * self.getHeight()
            if(max_patch_area != None):
                areaMax = max_patch_area

            areaMin = areaMin / (self.getResolution()**2)
            areaMax = areaMax / (self.getResolution()**2)

            for i in range(1, nPatches+1):
                patchCoordinates = numpy.nonzero(labelField==i)

                # Check patch size:
                nPatchPixels = len(patchCoordinates[0])
                if nPatchPixels < areaMin or nPatchPixels > areaMax:  # Black out areas that are too small or too big for a circle
                    nCleaned += 1
                    continue
                
                coordinatesX = patchCoordinates[1]
                coordinatesY = patchCoordinates[0]

                left  = numpy.amin(coordinatesX)
                right = numpy.amax(coordinatesX)
                top   = numpy.amin(coordinatesY)
                bottom= numpy.amax(coordinatesY)

                if remove_border_patches:   
                    if((left==0) or (top==0) or (right==self.getWidth()-1) or (bottom==self.getHeight()-1)):
                        nCleaned += 1
                        continue

                # An ideal circle should have an aspect ratio of 1:
                if aspect_ratio_tolerance != None:
                    aspectRatio = 0
                    if(top != bottom):
                        aspectRatio = abs(right-left) / abs(bottom-top)

                    if abs(1-aspectRatio) > aspect_ratio_tolerance:  # This is not a circle
                        nCleaned += 1
                        log("Aspect ratio {ar:.3f} doesn't meet aspect ratio tolerance |1-AR|={tolerance:.3f}".format(ar=aspectRatio, tolerance=aspect_ratio_tolerance))
                        continue

                # Add patch center as its coordinate:
                patchGeometry.append(((right+left)/2.0, (bottom+top)/2.0, right-left, bottom-top))

                self._px[patchCoordinates] = 1
                nRemaining += 1

        return nPatches, nCleaned, nRemaining, patchGeometry

    def fitCircle(self):
        # Linear least squares method by:
        # I. D. Coope,
        # Circle Fitting by Linear and Nonlinear Least Squares,
        # Journal of Optimization Theory and Applications, 1993, Volume 76, Issue 2, pp 381-388
        # https://doi.org/10.1007/BF00939613

        coordinates = numpy.nonzero(self._px)
        circlePixelsX = coordinates[1]
        circlePixelsY = coordinates[0]
        nPoints = len(circlePixelsX)
        circlePixels1 = numpy.ones(nPoints)

        # Create the matrix B for the system of linear equations:
        matrixB = numpy.array((circlePixelsX, circlePixelsY, circlePixels1))
        matrixB = matrixB.transpose()

        # linear equation to optimize:
        # matrix B * result = vector d
        d = []
        for i in range(nPoints):
            d.append(circlePixelsX[i]**2 + circlePixelsY[i]**2)

        vectorD = numpy.array(d)

        results, residuals, rank, s = numpy.linalg.lstsq(matrixB, vectorD, rcond=None)

        centerX = (results[0] / 2.0)
        centerY = (results[1] / 2.0)
        radius  = math.sqrt(results[2] + centerX**2 + centerY**2)

        # Calculate deviation statistics:
        differenceSum = 0
        minDifference = 99999
        maxDifference = 0
        for i in range(nPoints):
            diff = abs(radius  -  math.sqrt((centerX - circlePixelsX[i])**2 + (centerY - circlePixelsY[i])**2))
            differenceSum += diff

            if minDifference > diff:
                minDifference = diff

            if maxDifference < diff:
                maxDifference = diff

        meanDifference = differenceSum / nPoints

        return centerX, centerY, radius, meanDifference, minDifference, maxDifference

    def intensityFunction2D(self, x, I0, mu, R, x0):   # Lambert-Beer-Law for ball intensity, to fit.
        radicand = numpy.power(R,2) - numpy.power((x-x0),2)
        
        # Avoid root of negative numbers
        radicand[radicand < 0] = 0   

        # Huge radicands lead to exp()->0, therefore avoid huge exponentiation:
        radicand[radicand > (1400*1400)] = (1400*1400)

        result = I0*numpy.exp(-2.0*mu*numpy.sqrt(radicand))

        return result

    def intensityFunction3D(self, coord, I0, mu, R, x0, y0):   # Lambert-Beer-Law for ball intensity, to fit.
        if len(coord) == 2:
            (x, y) = coord

            radicand = numpy.power(R,2) - numpy.power((x-x0),2) - numpy.power((y-y0),2)
            
            # Avoid root of negative numbers
            radicand[radicand < 0] = 0   

            # Huge radicands lead to exp()->0, therefore avoid huge exponentiation:
            radicand[radicand > (1400*1400)] = (1400*1400)

            result = I0 * numpy.exp(-2.0*mu*numpy.sqrt(radicand))
            
            return result
        else:
            raise Exception("3D Intensity fit function expects a tuple (x,y) for coordinates.")

    def fitIntensityProfile(self, axis="x", initI0=None, initMu=0.003, initR=250, initX0=None, avgLines=5):
        yData = 0
        xdata = 0
        if initI0 == None:
            initI0 = self.getMax()   # Hoping that a median has been applied before.

        if axis == "x":
            if initX0 == None:
                initX0 = self.getWidth() / 2

            startLine = int((self.getHeight() / 2) - math.floor(avgLines/2))
            stopLine  = int((self.getHeight() / 2) + math.floor(avgLines/2))

            # Accumulate intensity profile along 'avgLines' lines around the center line:
            yData = numpy.zeros(self.getWidth(), dtype=self.getInternalDataType())
            for l in range(startLine, stopLine+1):
                yData += self._px[l,:]

            xData = numpy.linspace(0, self.getWidth()-1, self.getWidth())

        elif axis == "y":
            if initX0 == None:
                initX0 = self.getHeight() / 2

            startLine = int((self.getWidth() / 2) - math.floor(avgLines/2))
            stopLine  = int((self.getWidth() / 2) + math.floor(avgLines/2))

            # Accumulate intensity profile along 'avgLines' lines around the center line:
            yData = numpy.zeros(self.getHeight(), dtype=self.getInternalDataType())
            for l in range(startLine, stopLine+1):
                yData += self._px[:,l]

            xData = numpy.linspace(0, self.getHeight()-1, self.getHeight())

        else:
            raise Exception("projectionImage::fitIntensityProfile() needs profile direction to be 'x' or 'y'.")

        yData = yData / int(avgLines)   # average intensity profile
        firstGuess = (initI0, initMu, initR, initX0)

        try:
            optimalParameters, covariances = optimize.curve_fit(self.intensityFunction2D, xData, yData, p0=firstGuess)
        except Exception:
            optimalParameters = (None, None, None, None)


        fittedI0 = optimalParameters[0]
        fittedMu = optimalParameters[1]
        fittedR  = optimalParameters[2]
        fittedX0 = optimalParameters[3]

        return fittedI0, fittedMu, fittedR, fittedX0

class ImageStack:
    """ Specify an image stack from a single file (RAW chunk) or
        a collection of single 2D RAW or TIFF files. """

    def __init__(self, filePattern=None, width=None, height=None, dataType=None, byteOrder=None, rawFileHeaderSize=0, rawImageHeaderSize=0, slices=None, startNumber=0, flipByteOrder=False):
        self._files = ImageFile(filePattern, dataType, byteOrder, flipByteOrder)

        self._width       = width
        self._height      = height
        self._nSlices     = slices   # number of slices in stack
        self._startNumber = startNumber

        # A RAW chunk can contain an overall file header, and
        # each image in the stack can contain an image header.
        self._rawFileHeaderSize = rawFileHeaderSize
        self._rawImageHeaderSize = rawImageHeaderSize

        self._isVolumeChunk = False    # Is this a volume chunk or is a file list provided?

        self._fileList = []
        self._fileNumbers = []   # store original stack number in file name

    def nSlices(self):
        return self._nSlices

    def isVolumeChunk(self):
        return self._isVolumeChunk

    def setVolumeChunk(self, isVolumeChunk):
        self._isVolumeChunk = isVolumeChunk

    def getFileByteOrder(self):
        return self._files.getByteOrder()

    def setFileByteOrder(self, byteOrder):
        self._files.setByteOrder(byteOrder)

    def getFileDataType(self):
        return self._files.getDataType()

    def setFileDataType(self, dataType):
        self._files.setDataType(dataType)

    def doFlipByteOrder(self):
        return self._files.doFlipByteOrder()

    def setFlipByteOrder(self, flipByteOrder):
        self._files.setFlipByteOrder(flipByteOrder)

    def fileStackInfo(self, filenameString):
        """ Split file pattern into lead & trail text, number of expected digits. """
        if '%' in filenameString:
            # A % sign in the provided file pattern indicates an image stack: e.g. %04d
            percentagePosition = filenameString.find("%")

            numberStart = percentagePosition + 1
            numberStop  = filenameString.find("d", percentagePosition)

            leadText  = ""
            if(percentagePosition > 0):
                leadText = filenameString[:percentagePosition]

            trailText = ""
            if((numberStop+1) < len(filenameString)):
                trailText = filenameString[(numberStop+1):]

            if(numberStop > numberStart):
                numberString = filenameString[numberStart:numberStop]
                if(numberString.isdigit()):
                    nDigitsExpected = int(numberString)
                    return leadText, trailText, nDigitsExpected
                else:
                    raise Exception("Image stack pattern is wrong. The wildcard for sequential digits in a filename must be %, followed by number of digits, followed by d, e.g. %04d")
            else:
                raise Exception("Image stack pattern is wrong. The wildcard for sequential digits in a filename must be %, followed by number of digits, followed by d, e.g. %04d")

        return filenameString, "", 0

    def buildStack(self):
        """ Build list of files that match given file name pattern. """
        self._fileList = []
        self._fileNumbers = []

        # Treat projection files
        inFilePattern = self._files.getFilename()
        inputFolder  = os.path.dirname(inFilePattern)
        projBasename = os.path.basename(inFilePattern)

        if inputFolder == "" or inputFolder == None:
            inputFolder = "."

        # Check if an image stack is provided:
        if('%' not in inFilePattern):
            self._fileList.append(inFilePattern)

            if(isTIFF(inFilePattern)):  # treat as single TIFF projection            
                self._isVolumeChunk = False
                testImage = Image(inFilePattern)
                testImage.read()
                self._width    = testImage.getWidth()
                self._height   = testImage.getHeight()
                self._nSlices  = 1
                self._files.setDataType(testImage._inputFile.getDataType())
            else:  # treat as raw chunk
                if (self._width != None) and (self._height != None):
                    if (self._files.getDataType() != None):
                        if os.path.isfile(inFilePattern):
                            self._isVolumeChunk = True

                            if (self._nSlices == None):
                                # Determine number of slices.
                                fileSizeInBytes = os.path.getsize(inFilePattern)
                                dataSizeInBytes = fileSizeInBytes - self._rawFileHeaderSize
                                bytesPerImage = self._rawImageHeaderSize + self._width * self._height * self._files.getDataType().itemsize

                                if (dataSizeInBytes >= bytesPerImage):
                                    if (dataSizeInBytes % bytesPerImage) == 0:
                                        self._nSlices = int(dataSizeInBytes / bytesPerImage)
                                        log("{} slices found in raw chunk.".format(self._nSlices))
                                    else:
                                        raise Exception("The raw chunk data size ({} bytes, without general file header) is not divisible by the calculated size of a single image ({} bytes, including image header). Therefore, the number of slices cannot be determined. {}".format(dataSizeInBytes, bytesPerImage, inFilePattern))
                                else:
                                    raise Exception("The raw chunk data size ({} bytes, without general file header) is smaller than the calculated size of a single image ({} bytes, including image header). {}".format(dataSizeInBytes, bytesPerImage, inFilePattern))
                        else:
                            raise Exception("File not found: {}".format(inFilePattern))
                    else:
                        raise Exception("Please provide the data type of the raw chunk.")
                else:
                    raise Exception("Please provide width and height (in pixels) of the raw chunk.")
        else:
            # A % sign in the provided file pattern indicates an image stack: e.g. %04d
            leadText, trailText, nDigitsExpected = self.fileStackInfo(projBasename)

            # Get list of files in input folder:
            fileList = os.listdir(inputFolder)
            fileList.sort()

            nImported = 0

            for f in fileList:
                file = inputFolder + "/" + f
                if os.path.isfile(file):
                    # Check if filename matches pattern:
                    if(f.startswith(leadText) and f.endswith(trailText)):
                        digitText = f[len(leadText):-len(trailText)]
                        if(digitText.isdigit() and len(digitText)==nDigitsExpected):
                            # Pattern matches.
                            n = int(digitText)
                            if n >= self._startNumber:
                                self._fileList.append(file)
                                self._fileNumbers.append(n)

                                nImported += 1
                                if nImported == self._nSlices:
                                    break
                        else:
                            continue
                    else:
                        continue

            self._nSlices = len(self._fileList)

            if self._nSlices > 0:
                if isTIFF(self._fileList[0]):
                    testImage = Image(self._fileList[0])
                    testImage.read()
                    self._width    = testImage.getWidth()
                    self._height   = testImage.getHeight()
                    self._files.setDataType(testImage._inputFile.getDataType())
                

    def getFilename(self, index=None):
        if index != None:
            if self._isVolumeChunk:
                if len(self._fileList) > 0:
                    return self._fileList[0]
                else:
                    return None
            else:
                if len(self._fileList) > index:
                    return self._fileList[index]
                else:
                    return None
        else:
            return self._files.getFilename()

    def getFileBasename(self, index=None):
        if index != None:
            if self._isVolumeChunk:
                if len(self._fileList) > 0:
                    return os.path.basename(self._fileList[0])
                else:
                    return None
            else:
                if len(self._fileList) > index:
                    return os.path.basename(self._fileList[index])
                else:
                    return None
        else:
            return self._files.getFileBasename()

    def setFilename(self, filename):
        self._files.setFilename(filename)

    def getImage(self, index, outputFile=None):
        """ Read and return image at position 'index' within the stack. """
        if index >= 0:
            if not self._isVolumeChunk:  # read single image file from stack:
                if len(self._fileList) > index:
                    filename = self._fileList[index]
                    file = ImageFile(filename=filename, dataType=self.getFileDataType(), byteOrder=self.getFileByteOrder(), flipByteOrder=self.doFlipByteOrder())

                    img = Image(file, outputFile)
                    if isTIFF(filename):
                        img.read()
                    else:
                        img.readRAW(self._width, self._height, 0, self.getFileDataType(), self.getFileByteOrder(), self._rawFileHeaderSize, self._rawImageHeaderSize)
                    return img
                else:
                    raise Exception("The requested slice nr. {} is out of bounds, because only {} image files were found.".format(index, len(self._fileList)))
            else:  # read slice from volume chunk
                if len(self._fileList) > 0:
                    file = self._fileList[0]
                    img = Image(file, outputFile)
                    if isTIFF(file):
                        raise Exception("Cannot treat 3D TIFFs.")
                    else:
                        img.readRAW(self._width, self._height, index, self.getFileDataType(), self.getFileByteOrder(), self._rawFileHeaderSize, self._rawImageHeaderSize)
                        return img
                else:
                    raise Exception("No image file specified to be loaded.")
        else:
            raise Exception("Negative slice numbers do not exists. {} requested.".format(index))

    def getMeanImage(self, outputFile=None):
        """ Calculate the mean of all image files. """
        if self.nSlices() > 0:
            if self.nSlices() > 1:
                sumImg = self.getImage(0, outputFile)
                for i in range(1, self.nSlices()):
                    sumImg.addImage(self.getImage(i, outputFile))

                sumImg.divide(self.nSlices())
                return sumImg
            else:
                return self.getImage(0, outputFile)
        else:
            return None
