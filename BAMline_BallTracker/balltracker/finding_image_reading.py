#How to read one hdf5-file instead of tiff-files


example_run.py / GUI.py
    inputFiles = ImageStack(filePattern= path_in + namepart + "%4d.tif", startNumber = startNumberProj, slices = numberProj)
    flatFields = ImageStack(filePattern= path_in + namepart + "%4d.tif", startNumber = startNumberFFs, slices = numberFFs)

    seq = ballSequence(inputFileStack=inputFiles, outputFolder=outputFolder, darkFileStack=darkFields, flatFileStack=flatFields)


ball.py
    class ballSequence:
        """ Manages the ball image sequence to find the trajectory and fit the trajectory parameters. """

        def __init__(self, inputFileStack=None, outputFolder=None, darkFileStack=None, flatFileStack=None):

            self.setInputFileStack(inputFileStack)
            self.setFlatFileStack(flatFileStack)

        def setInputFileStack(self, inputFileStack):
            self._inputFileStack = createImageStack(inputFileStack)

        def setFlatFileStack(self, flatFileStack):
            self._flatFileStack = createImageStack(flatFileStack)
            if isinstance(self._flatFileStack, ImageStack):
                self._applyRefs = True
            else:
                self._applyRefs = False


image.py
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


    class ImageStack:
        """ Specify an image stack from a single file (RAW chunk) or
            a collection of single 2D RAW or TIFF files. """

        def __init__(self, filePattern=None, width=None, height=None, dataType=None, byteOrder=None, rawFileHeaderSize=0, rawImageHeaderSize=0, slices=None, startNumber=0, flipByteOrder=False):
            self._files = ImageFile(filePattern, dataType, byteOrder, flipByteOrder)
