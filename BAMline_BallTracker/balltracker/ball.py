# -*- coding: UTF-8 -*-
import numpy, os, copy, math, re, time

# Scipy:
# 'ndimage' class for image processing
# 'optimize' class for intensity fit
# 'signal' class for drift analysis using FFT Convolution
from scipy import ndimage, optimize, stats, signal

# Matplotlib for graphics display.
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# The 'feature' package from scikit-image,
# only needed for Canny edge detection, when used instead of Sobel.
from skimage import feature

# Multiprocessing
from multiprocessing import Pool

from .image import *
from .general import *

class ballDataset:
    """ One coordinate in the list of ball coordinates,
        mostly to have one object that can be returned by
        a multiprocessing thread. """

    def __init__(self, filename=None, index=0, angle=None, x=None, y=None, radius=None, meanDev=None, maxDev=None):
        self._filename = None
        self._phi      = None   # Angular position during the rotation, in deg.
        self._projIndex = 0

        self._x        = None   # Center X, from fit of circle function
        self._y        = None   # Center Y, from fit of circle function
        self._xInt     = None   # Center X, from fit of intensity profile
        self._yInt     = None   # Center Y, from fit of intensity profile

        self._radius   = None   # Radius from fit of circle function.
        self._radiusInt= None   # Radius from fit of intensity profile.

        self._meanDeviation = None
        self._maxDeviation  = None

        self._accepted = False  # Did ball meet all criteria?
    
        # Properties to display an image after processing:
        self._displayX      = 0   # center X, on binned/cropped image
        self._displayY      = 0   # center Y, on binned/cropped image
        self._displayRadius = 0   # radius, on binned/cropped image

        self.setFilename(filename)
        self.setAngle(angle, unit='deg')
        self.setProjectionIndex(index)
        self.setX(x)
        self.setY(y)
        self.setRadius(radius)
        self.setMeanDev(meanDev)
        self.setMaxDev(maxDev)

    def setFilename(self, filename):
        self._filename = filename

    def setAngle(self, phi, unit='deg'):
        if unit == 'deg':
            self._phi = phi
        elif unit == 'rad':
            self._phi = rad2deg(phi)
        else:
            raise Exception("Angle unit must be 'deg' or 'rad'.")

    def setProjectionIndex(self, index):
        self._projIndex = index

    def setX(self, x):
        self._x = x

    def setY(self, y):
        self._y = y

    def setXint(self, xint):
        self._xInt = xint

    def setYint(self, yint):
        self._yInt = yint

    def setRadius(self, radius):
        self._radius = radius

    def setRadiusInt(self, radiusInt):
        self._radiusInt = radiusInt

    def setMeanDev(self, meanDev):
        self._meanDeviation = meanDev

    def setMaxDev(self, maxDev):
        self._maxDeviation = maxDev

    def setAccepted(self, accepted=False):
        self._accepted = accepted


    def getFilename(self):
        return self._filename

    def getAngle(self, unit='deg'):
        return deg2x(self._phi, unit)

    def getProjectionIndex(self):
        return self._projIndex

    def getX(self):
        return self._x

    def getY(self):
        return self._y

    def getXint(self):
        return self._xInt

    def getYint(self):
        return self._yInt

    def getRadius(self):
        return self._radius

    def getRadiusInt(self):
        return self._radiusInt

    def getMeanDev(self):
        return self._meanDeviation

    def getMaxDev(self):
        return self._maxDeviation


    # Handle display parameters:
    def setDisplayX(self, dispx):
        self._displayX = dispx

    def getDisplayX(self):
        return self._displayX

    def setDisplayY(self, dispy):
        self._displayY = dispy

    def getDisplayY(self):
        return self._displayY

    def setDisplayRadius(self, disprad):
        self._displayRadius = disprad

    def getDisplayRadius(self):
        return self._displayRadius  

    def isAccepted(self, mode="circle"):
        if mode=="circle":
            return self._accepted
        elif mode=="intensity":
            if (self.getXint() == None) or (self.getYint() == None) or (self.getRadiusInt() == None):
                return False
            else:
                return True

class ballWave:
    """ Handles wave data (x, y or R) and its fitting. """

    def __init__(self):
        self._angles = []
        self._data   = []

        self._isSecondOrderWave = False

        # Sinus fit results:  A*sin(phi + phi0) + B*sin(omega*phi +theta0) + n
        self._A         = 0
        self._phi0      = 0
        self._n         = 0

        self._B         = 0
        self._omega     = 0
        self._theta0    = 0

        self._errA      = 0
        self._errPhi0   = 0
        self._errN      = 0

        self._errB      = 0
        self._errOmega  = 0
        self._errTheta0 = 0

        # Parameters for matrix transformation fit (wobble):
        self._mt_R             = 0  # trajectory radius, this is constant
        self._mt_L             = 0  # axis length
        self._mt_phi0          = 0  # sample phase
        self._mt_alpha         = 0  # tilt around X axis
        self._mt_beta          = 0  # tilt around Y axis
        self._mt_W             = 0  # Wobble amplitude
        self._mt_omega         = 0  # Wobble frequency
        self._mt_omega0        = 0  # Wobble phase

        self._mt_err_R         = 0
        self._mt_err_L         = 0
        self._mt_err_phi0      = 0
        self._mt_err_alpha     = 0
        self._mt_err_beta      = 0
        self._mt_err_W         = 0
        self._mt_err_omega     = 0
        self._mt_err_omega0    = 0

        self._fitSuccessful                = False
        self._matrixTransformFitSuccessful = False

    def clear(self):
        self.__init__()

    def setAngles(self, angles):
        self._angles = angles

    def getAngles(self):
        return self._angles

    def setData(self, data):
        self._data = data

    def getData(self):
        return self._data

    def addDataset(self, phi, x):
        self._angles.append(phi)
        self._data.append(x)

    def length(self):
        if len(self._angles) == len(self._data):
            return len(self._angles)
        else:
            raise Exception("ballWave: number of angles ({}) doesn't match number of data points ({}).".format(len(self._angles), len(self._data)))

    def getPhi(self, i, unit='rad'):
        if i<len(self._angles):
            return rad2x(self._angles[i], unit=unit)
        else:
            raise Exception("ballWave: Requested index {} exceeds number of stored angles ({}).".format(i), len(self._angles))

    def getPoint(self, i):
        if i<len(self._data):
            return self._data[i]
        else:
            raise Exception("ballWave: Requested index {} exceeds number of stored data points ({}).".format(i), len(self._data))

    def getA(self):
        return self._A

    def getPhi0(self):
        return self._phi0

    def getN(self):
        return self._n

    def getB(self):
        return self._B

    def getOmega(self):
        return self._omega

    def getTheta0(self):
        return self._theta0

    def getA_err(self):
        return self._errA

    def getPhi0_err(self):
        return self._errPhi0

    def getN_err(self):
        return self._errN

    def getB_err(self):
        return self._errB

    def getOmega_err(self):
        return self._errOmega

    def getTheta0_err(self):
        return self._errTheta0

    def setMTR(self, R):
        self._mt_R = R

    def getMTR(self):
        return self._mt_R

    def getMTR_err(self):
        return self._mt_err_R

    def getMTL(self):
        return self._mt_L

    def getMTL_err(self):
        return self._mt_err_L

    def getMTphi0(self):
        return self._mt_phi0

    def getMTphi0_err(self):
        return self._mt_err_phi0

    def setMTalpha(self, alpha):
        self._mt_alpha = alpha

    def getMTalpha(self):
        return self._mt_alpha

    def getMTalpha_err(self):
        return self._mt_err_alpha

    def setMTbeta(self, beta):
        self._mt_beta = beta

    def getMTbeta(self):
        return self._mt_beta

    def getMTbeta_err(self):
        return self._mt_err_beta

    def getMTW(self):
        return self._mt_W

    def getMTW_err(self):
        return self._mt_err_W

    def getMTomega(self):
        return self._mt_omega

    def getMTomega_err(self):
        return self._mt_err_omega

    def getMTomega0(self):
        return self._mt_omega0

    def getMTomega0_err(self):
        return self._mt_err_omega0

    def sinusFunction(self, phi, A, phi0, n):
        return A * numpy.sin(phi + phi0) + n

    def sinusFunction2(self, phi, A, phi0, B, omega, theta0, n):
        return A*numpy.sin(phi+phi0) + B*numpy.sin(omega*phi+theta0) + n

    def ballTrajectoryX(self, phi, L, phi0, W, omega, omega0):
        return self._mt_R*numpy.cos(phi+phi0) + L*(W*numpy.sin(omega*phi + omega0)*numpy.cos(phi) + self._mt_beta)

    def ballTrajectoryZ(self, phi, L, phi0, W, omega, omega0):
        return self._mt_R*(self._mt_alpha*numpy.sin(phi+phi0) - self._mt_beta*numpy.cos(phi+phi0) - W*numpy.sin(omega*phi+omega0)*numpy.cos(phi0)) + L

    def getFitValue(self, phi):
        return self.sinusFunction(phi=phi, A=self._A, phi0=self._phi0, n=self._n)

    def getFitValue2(self, phi):
        return self.sinusFunction2(
            phi=phi,
            A=self._A,
            phi0=self._phi0,
            B=self._B,
            omega=self._omega,
            theta0=self._theta0,
            n=self._n
            )

    def getTrajectoryValueX(self, phi):
        return self.ballTrajectoryX(
            phi=phi,
            L=self._mt_L,
            phi0=self._mt_phi0,
            W=self._mt_W,
            omega=self._mt_omega,
            omega0=self._mt_omega0
            )

    def getTrajectoryValueZ(self, phi):
        return self.ballTrajectoryZ(
            phi=phi,
            L=self._mt_L,
            phi0=self._mt_phi0,
            W=self._mt_W,
            omega=self._mt_omega,
            omega0=self._mt_omega0
            )

    def isSecondOrderWave(self):
        return self._isSecondOrderWave

    def fitSuccessful(self):
        return self._fitSuccessful

    def matrixTransformFitSuccessful(self):
        return self._matrixTransformFitSuccessful

    def fitSinus(self, initA=None, initPhi0=0, initN=2000):
        firstGuess = (initA, initPhi0, initN)

        optimalParameters, covariances = optimize.curve_fit(self.sinusFunction, self._angles, self._data, p0=firstGuess)

        self._A         = optimalParameters[0]
        self._phi0      = optimalParameters[1]
        self._B         = 0
        self._omega     = 0
        self._theta0    = 0
        self._n         = optimalParameters[2]

        self._errA      = math.sqrt(covariances[0][0])
        self._errPhi0   = math.sqrt(covariances[1][1])
        self._errB      = 0
        self._errOmega  = 0
        self._errTheta0 = 0
        self._errN      = math.sqrt(covariances[2][2])

        # Do not allow negative amplitudes. Instead, shift by Pi:
        if self._A < 0:
            self._A = -self._A
            if self._phi0 > math.pi:
                self._phi0 -= math.pi
            else:
                self._phi0 += math.pi

        self._isSecondOrderWave = False
        self._fitSuccessful = True
        return self._A, self._phi0, self._n, self._errA, self._errPhi0, self._errN

    def fitSinus2(self, initA=None, initPhi0=0, initB=1, initOmega=2, initTheta0=0, initN=2000):
        firstGuess = (initA, initPhi0, initB, initOmega, initTheta0, initN)

        optimalParameters, covariances = optimize.curve_fit(self.sinusFunction2, self._angles, self._data, p0=firstGuess)

        self._A         = optimalParameters[0]
        self._phi0      = optimalParameters[1]
        self._B         = optimalParameters[2]
        self._omega     = optimalParameters[3]
        self._theta0    = optimalParameters[4]
        self._n         = optimalParameters[5]

        self._errA      = math.sqrt(covariances[0][0])
        self._errPhi0   = math.sqrt(covariances[1][1])
        self._errB      = math.sqrt(covariances[2][2])
        self._errOmega  = math.sqrt(covariances[3][3])
        self._errTheta0 = math.sqrt(covariances[4][4])
        self._errN      = math.sqrt(covariances[5][5])

        # Do not allow negative amplitudes. Instead, shift by Pi:
        if self._A < 0:
            self._A = -self._A
            if self._phi0 > math.pi:
                self._phi0 -= math.pi
            else:
                self._phi0 += math.pi

        if self._B < 0:
            self._B = -self._B
            if self._theta0 > math.pi:
                self._theta0 -= math.pi
            else:
                self._theta0 += math.pi

        self._isSecondOrderWave = True
        self._fitSuccessful = True
        return self._A, self._phi0, self._B, self._omega, self._theta0, self._n, self._errA, self._errPhi0, self._errB, self._errOmega, self._errTheta0, self._errN

    def fitTrajectoryX(self, initL, initPhi0, initW, initOmega, initOmega0):
        firstGuess = (initL, initPhi0, initW, initOmega, initOmega0)

        optimalParameters, covariances = optimize.curve_fit(self.ballTrajectoryX, self._angles, self._data, p0=firstGuess)

        self._mt_L      = optimalParameters[0]
        self._mt_phi0   = optimalParameters[1]
        self._mt_W      = optimalParameters[2]
        self._mt_omega  = optimalParameters[3]
        self._mt_omega0 = optimalParameters[4]

        self._mt_err_L      = math.sqrt(covariances[0][0])
        self._mt_err_phi0   = math.sqrt(covariances[1][1])
        self._mt_err_W      = math.sqrt(covariances[2][2])
        self._mt_err_omega  = math.sqrt(covariances[3][3])
        self._mt_err_omega0 = math.sqrt(covariances[4][4])

        # Do not allow negative amplitudes. Instead, shift by Pi:
        if self._mt_W < 0:
            self._mt_W = -self._mt_W
            if self._mt_omega0 > math.pi:
                self._mt_omega0 -= math.pi
            else:
                self._mt_omega0 += math.pi

        self._matrixTransformFitSuccessful = True

        return self._mt_L, self._mt_phi0, self._mt_W, self._mt_omega, self._mt_omega0, self._mt_err_L, self._mt_err_phi0, self._mt_err_W, self._mt_err_omega, self._mt_err_omega0

    def fitTrajectoryZ(self, initL, initPhi0, initW, initOmega, initOmega0):
        firstGuess = (initL, initPhi0, initW, initOmega, initOmega0)

        optimalParameters, covariances = optimize.curve_fit(self.ballTrajectoryZ, self._angles, self._data, p0=firstGuess)

        self._mt_L      = optimalParameters[0]
        self._mt_phi0   = optimalParameters[1]
        self._mt_W      = optimalParameters[2]
        self._mt_omega  = optimalParameters[3]
        self._mt_omega0 = optimalParameters[4]

        self._mt_err_L      = math.sqrt(covariances[0][0])
        self._mt_err_phi0   = math.sqrt(covariances[1][1])
        self._mt_err_W      = math.sqrt(covariances[2][2])
        self._mt_err_omega  = math.sqrt(covariances[3][3])
        self._mt_err_omega0 = math.sqrt(covariances[4][4])

        # Do not allow negative amplitudes. Instead, shift by Pi:
        if self._mt_W < 0:
            self._mt_W = -self._mt_W
            if self._mt_omega0 > math.pi:
                self._mt_omega0 -= math.pi
            else:
                self._mt_omega0 += math.pi

        self._matrixTransformFitSuccessful = True

        return self._mt_L, self._mt_phi0, self._mt_W, self._mt_omega, self._mt_omega0, self._mt_err_L, self._mt_err_phi0, self._mt_err_W, self._mt_err_omega, self._mt_err_omega0


class ballTrajectory:
    """ Handles the data, fitting and results of a trajectory fit to get the axis parameters. """

    def __init__(self, name="Trajectory", beamAxisName="z", rowAxisName="x", colAxisName="y"):
        self._beamAxis = beamAxisName
        self._rowAxis = rowAxisName
        self._colAxis = colAxisName

        # Data for plotting the fit functions:
        self._fitFunctionAngles = []
        self._rangeStart        = 0
        self._rangeStop         = 0
        self._fitSecondOrder    = True

        # Points where ellipse crosses x and y axis of its own coordinate system:
        self._axisPoints        = []

        # Coordinates of apex points of the ellipse, in centered detector coordinate system:
        self._apexPoints        = []

        # Ellipse points of intersection with centered detector coordinate system:
        self._x1intersect = 0
        self._x2intersect = 0
        self._y1intersect = 0
        self._y2intersect = 0

        # Axis results:
        self._tiltX = 0
        self._tiltY = 0
        self._centerOfRotation = 0
        self._trajectoryRadius = 0  # 3D circular trajectory radius

        self._xWave = ballWave()
        self._yWave = ballWave()
        self._rWave = ballWave()  # radius results

        self._name  = ""
        self.setName(name)

    def clear(self):
        self._xWave.clear()
        self._yWave.clear()
        self.__init__(name=self._name, beamAxisName=self._beamAxis, rowAxisName=self._rowAxis, colAxisName=self._colAxis)

    def setName(self, name):
        self._name = name

    def getName(self):
        return self._name

    def setTiltX(self, tiltX):
        self._tiltX = tiltX

    def getTiltX(self):
        return self._tiltX

    def setTiltY(self, tiltY):
        self._tiltY = tiltY

    def getTiltY(self):
        return self._tiltY

    def setCenterOfRotation(self, cor):
        self._centerOfRotation = cor

    def getCenterOfRotation(self):
        return self._centerOfRotation

    def setTrajectoryRadius(self, radius):
        self._trajectoryRadius = radius

    def getTrajectoryRadius(self):
        return self._trajectoryRadius

    def add(self, angle=None, x=None, y=None, r=None):
        if (x != None) and (y != None) and (angle != None) and (r != None):
            self._xWave.addDataset(float(angle), float(x))
            self._yWave.addDataset(float(angle), float(y))
            self._rWave.addDataset(float(angle), float(r))

    def getStartAngle(self, unit='rad'):
        if self._xWave.length() > 0:
            angle = self._xWave.getPhi(0)
            return rad2x(angle, unit)
        else:
            return None     

    def getStopAngle(self, unit='rad'):
        if self._xWave.length() > 0:
            angle = self._xWave.getPhi(-1)
            return rad2x(angle, unit)
        else:
            return None

    def xWave(self):
        return self._xWave

    def yWave(self):
        return self._yWave

    def rWave(self):
        return self._rWave

    def fitSuccessful(self):
        return (self._xWave.fitSuccessful()) and (self._yWave.fitSuccessful())

    def distance(self, x0, y0, x1, y1):
        return numpy.sqrt((x1-x0)**2 + (y1-y0)**2)

    def numberOfDatasets(self):
        nx = self._xWave.length()
        ny = self._yWave.length()
        nr = self._rWave.length()

        if (nx == ny) and (ny == nr):
            return nx
        else:
            raise Exception("Trajectory data points: number of list elements for {rowaxis} ({nx}), {colaxis} ({ny}) and radii ({nr}) does not match.".format(rowaxis=self._rowAxisName, nx=nx, colaxis=self._colAxisName, ny=ny, nr=nr))

    def formatGeometryResults(self, extended=False):
        output  = self.getName() + ":\n"
        output += "=================================\n"
        output += "Tilt around {rowaxis}:      {rad:.8f} rad = {deg:.7f} deg\n".format(rowaxis=self._rowAxis, rad=self.getTiltX(), deg=rad2deg(self.getTiltX()))
        output += "Tilt around {beamaxis}:      {rad:.8f} rad = {deg:.7f} deg\n".format(beamaxis=self._beamAxis, rad=self.getTiltY(), deg=rad2deg(self.getTiltY()))
        output += "Center of rotation: {:.3f} px +- {:.3f} px\n".format(self.getCenterOfRotation(), self._xWave.getN_err())
        output += "Trajectory radius:  {:.3f} px +- {:.3f} px\n".format(self.getTrajectoryRadius(), self._xWave.getA_err())

        if extended == True:
            output += "\nFit results for {rowaxis},{colaxis} = A*sin(phi + phi0) + B*sin(omega*phi + theta0) + n\n\n".format(rowaxis=self._rowAxis, colaxis=self._colAxis)

            # Row axis:
            output += "A_{}      = {:.8f} +- {:.8f}\n".format(self._rowAxis, self._xWave.getA(), self._xWave.getA_err())
            output += "phi0_{}   = {:.8f} +- {:.8f}\n".format(self._rowAxis, self._xWave.getPhi0(), self._xWave.getPhi0_err())
            output += "B_{}      = {:.8f} +- {:.8f}\n".format(self._rowAxis, self._xWave.getB(), self._xWave.getB_err())
            output += "omega_{}  = {:.8f} +- {:.8f}\n".format(self._rowAxis, self._xWave.getOmega(), self._xWave.getOmega_err())
            output += "theta0_{} = {:.8f} +- {:.8f}\n".format(self._rowAxis, self._xWave.getTheta0(), self._xWave.getTheta0_err())
            output += "n_{}      = {:.8f} +- {:.8f}\n\n".format(self._rowAxis, self._xWave.getN(), self._xWave.getN_err())

            # Col axis:
            output += "A_{}      = {:.8f} +- {:.8f}\n".format(self._colAxis, self._yWave.getA(), self._yWave.getA_err())
            output += "phi0_{}   = {:.8f} +- {:.8f}\n".format(self._colAxis, self._yWave.getPhi0(), self._yWave.getPhi0_err())
            output += "B_{}      = {:.8f} +- {:.8f}\n".format(self._colAxis, self._yWave.getB(), self._yWave.getB_err())
            output += "omega_{}  = {:.8f} +- {:.8f}\n".format(self._colAxis, self._yWave.getOmega(), self._yWave.getOmega_err())
            output += "theta0_{} = {:.8f} +- {:.8f}\n".format(self._colAxis, self._yWave.getTheta0(), self._yWave.getTheta0_err())
            output += "n_{}      = {:.8f} +- {:.8f}\n\n".format(self._colAxis, self._yWave.getN(), self._yWave.getN_err())

        return output

    def formatMatrixTransformationFitResults(self):
        output  = "Wobble from {colaxis} oscillations:\n".format(colaxis=self._colAxis)
        output += "---------------------------------\n"
        output += "Wobble Amplitude:   {:.8f} rad +- {:.7f} rad\n".format(self._yWave.getMTW(), self._yWave.getMTW_err())
        output += "                  = {:.8f} deg +- {:.7f} deg\n".format(rad2deg(self._yWave.getMTW()), rad2deg(self._yWave.getMTW_err()))
        output += "Wobble Frequency:   {:.3f} +- {:.3f}\n".format(self._yWave.getMTomega(), self._yWave.getMTomega_err())
        output += "Wobble Phase:       {:.3f} rad +- {:.3f} rad\n".format(self._yWave.getMTomega0(), self._yWave.getMTomega0_err())
        output += "Axis Length L:      {:.3f} px +- {:.3f} px\n".format(self._yWave.getMTL(), self._yWave.getMTL_err())
        output += "Sample Phase phi0:  {:.3f} rad +- {:.3f} rad\n".format(self._yWave.getMTphi0(), self._yWave.getMTphi0_err())

        return output

    def analyseTrajectory(self, secondOrder=True):
        self._fitSecondOrder = secondOrder
        if self.numberOfDatasets() > 2:
            angles = numpy.array(self._xWave.getAngles())
            xPositions = numpy.array(self._xWave.getData())
            yPositions = numpy.array(self._yWave.getData())

            initAx = (numpy.amax(xPositions) - numpy.amin(xPositions)) / 2
            initNx = (numpy.amax(xPositions) + numpy.amin(xPositions)) / 2

            if self._fitSecondOrder == False:
                Ax, phi0x, nx, errAx, errPhi0x, errNx = self._xWave.fitSinus(initA=initAx, initPhi0=0, initN=initNx)
            else:
                try:
                    Ax, phi0x, Bx, omegax, theta0x, nx, errAx, errPhi0x, errBx, errOmegax, errorThetax, errNx = self._xWave.fitSinus2(initA=initAx, initPhi0=0, initN=initNx)
                except Exception:
                    print("WARNING: Can't fit second order {rowaxis} sines. Retreat to first order.".format(rowaxis=self._rowAxis))
                    Ax, phi0x, nx, errAx, errPhi0x, errNx = self._xWave.fitSinus(initA=initAx, initPhi0=0, initN=initNx)

            initAy = (numpy.amax(yPositions) - numpy.amin(yPositions)) / 2
            initNy = (numpy.amax(yPositions) + numpy.amin(yPositions)) / 2

            if self._fitSecondOrder == False:
                Ay, phi0y, ny, errAy, errPhi0y, errNy = self._yWave.fitSinus(initA=initAy, initPhi0=0, initN=initNy)
            else:
                try:
                    Ay, phi0y, By, omegay, theta0y, ny, errAy, errPhi0y, errBy, errOmegay, errorThetay, errNy = self._yWave.fitSinus2(initA=initAy, initPhi0=0, initN=initNy)
                except Exception:
                    print("WARNING: Can't fit second order {colaxis} sines. Retreat to first order.".format(colaxis=self._colAxis))
                    Ay, phi0y, ny, errAy, errPhi0y, errNy = self._yWave.fitSinus(initA=initAy, initPhi0=0, initN=initNy)

            # Find intersections with coordinate system:
            self._y1intersect = self._yWave.getFitValue(-phi0x)
            self._y2intersect = self._yWave.getFitValue(math.pi-phi0x)

            self._x1intersect = self._xWave.getFitValue(-phi0y)
            self._x2intersect = self._xWave.getFitValue(math.pi-phi0y)

            turningPointPhi1 = -phi0x+math.pi/2
            turningPointPhi2 = -phi0x-math.pi/2
            turningPointX1 = self._xWave.getFitValue(turningPointPhi1)
            turningPointX2 = self._xWave.getFitValue(turningPointPhi2)
            turningPointY1 = self._yWave.getFitValue(turningPointPhi1)
            turningPointY2 = self._yWave.getFitValue(turningPointPhi2)

            self._axisPoints = ((turningPointX1, turningPointX2, nx, nx), (turningPointY1, turningPointY2, self._y1intersect, self._y2intersect))

            self._rangeStart = self.getStartAngle(unit='rad')
            self._rangeStop  = self.getStopAngle(unit='rad') + 0.1

            # Plot at least one full circle:
            if (self._rangeStop - self._rangeStart) <= (2*math.pi):
                self._rangeStop = self._rangeStart + 2*math.pi + 0.1

            # 100 samples per 2pi:
            nPointsInFitCurve = int(math.ceil((self._rangeStop - self._rangeStart) * 50 / math.pi))

            self._fitFunctionAngles = numpy.linspace(self._rangeStart, self._rangeStop, int(nPointsInFitCurve))

            # Find ellipse apex points:
            maxDistance = 0
            maxX0 = 0
            maxX1 = 0
            maxY0 = 0
            maxY1 = 0
            for phi in self._fitFunctionAngles:
                x0 = self._xWave.getFitValue(phi) - nx
                x1 = self._xWave.getFitValue(phi+math.pi) - nx
                y0 = self._yWave.getFitValue(phi) - ny
                y1 = self._yWave.getFitValue(phi+math.pi) - ny
                d  = self.distance(x0, y0, x1, y1)

                if d > maxDistance:
                    maxDistance = d
                    maxX0 = x0
                    maxX1 = x1
                    maxY0 = y0
                    maxY1 = y1

            R_3Dcircle = maxDistance/2
            self._apexPoints = ((maxX0+nx, maxX1+nx), (maxY0+ny, maxY1+ny))

            tiltX = (self._y2intersect-self._y1intersect) / (2*R_3Dcircle)
            alpha_tiltX = math.asin(tiltX)

            #ySlope = (numpy.amax(fitY) - numpy.amin(fitY)) / (numpy.amax(fitX) - numpy.amin(fitX))
            ySlope = (turningPointY2 - turningPointY1) / (turningPointX2 - turningPointX1)
            beta_tiltY = numpy.arctan(ySlope)

            print("Tilt {rowaxis}:".format(rowaxis=self._rowAxis))
            print("Rotation axis intersect 1 = {}".format(self._y1intersect))
            print("Rotation axis intersect 2 = {}".format(self._y2intersect))
            print("Radius 3D = {}".format(R_3Dcircle))

            print("Tilt {beamaxis}:".format(beamaxis=self._beamAxis))
            print("Turning Point {rowaxis}1 = {x1}".format(rowaxis=self._rowAxis, x1=turningPointX1))
            print("Turning Point {colaxis}1 = {y1}".format(colaxis=self._colAxis, y1=turningPointY1))
            print("Turning Point {rowaxis}2 = {x2}".format(rowaxis=self._rowAxis, x2=turningPointX2))
            print("Turning Point {colaxis}2 = {y2}".format(colaxis=self._colAxis, y2=turningPointY2))
            print("Apex Points: {}".format(self._apexPoints))

            self.setTiltX(alpha_tiltX)
            self.setTiltY(beta_tiltY)
            self.setCenterOfRotation(nx)
            self.setTrajectoryRadius(R_3Dcircle)

            print(self.formatGeometryResults())

            # Matrix Fit:
            """
            self._xWave.setMTR(R_3Dcircle)
            self._xWave.setMTalpha(alpha_tiltX)
            self._xWave.setMTbeta(beta_tiltY)
            self._yWave.setMTR(R_3Dcircle)
            self._yWave.setMTalpha(alpha_tiltX)
            self._yWave.setMTbeta(beta_tiltY)

            #try:
            #   mtL, mtPhi0, mtW, mtOmega, mtOmega0, errL, errMtPhi0, errMtW, errMtOmega, errMtOmega0 = self._xWave.fitTrajectoryX(initL=ny, initPhi0=0, initW=0.0002, initOmega=2, initOmega0=0)
            #except Exception:
            #   print("Can't fit X trajectory from matrix transformations.")

            try:
                mtL, mtPhi0, mtW, mtOmega, mtOmega0, errL, errMtPhi0, errMtW, errMtOmega, errMtOmega0 = self._yWave.fitTrajectoryZ(initL=ny, initPhi0=0, initW=0.0002, initOmega=2, initOmega0=0)
            except Exception:
                print("Can't fit Z trajectory from matrix transformations.")

            print(self.formatMatrixTransformationFitResults())
            """

        else:
            raise Exception("{} points of data are not enough for a successful axis geometry analysis.".format(self.numberOfDatasets()))

    def plotResults(self, displayPlot=True, savePlot=True, saveAs=None, interactive=False, pointColor='blue', curveColor='orange', resultColor='red'):
        if self.numberOfDatasets() > 2:
            plt.ioff()

            angles = numpy.array(self._xWave.getAngles())
            xPositions = numpy.array(self._xWave.getData())
            yPositions = numpy.array(self._yWave.getData())

            if self.fitSuccessful():
                fitX = self._xWave.getFitValue(self._fitFunctionAngles)
                fitY = self._yWave.getFitValue(self._fitFunctionAngles)

                if self._xWave.isSecondOrderWave():
                    fit2X = self._xWave.getFitValue2(self._fitFunctionAngles)

                if self._yWave.isSecondOrderWave():
                    fit2Y = self._yWave.getFitValue2(self._fitFunctionAngles)

                centerOfRotationX = numpy.array((self._rangeStart, self._rangeStop))
                centerOfRotationY = numpy.array((numpy.amin(yPositions), numpy.amax(yPositions)))
                centerOfRotation  = numpy.array((self.getCenterOfRotation(), self.getCenterOfRotation()))

                if self._xWave.matrixTransformFitSuccessful():
                    fitMTx = self._xWave.getTrajectoryValueX(self._fitFunctionAngles)

                if self._yWave.matrixTransformFitSuccessful():
                    fitMTz = self._yWave.getTrajectoryValueZ(self._fitFunctionAngles)

            # Create labels for angle axis:
            phiTicks  = []
            phiLabels = []
            nSubticks = 2
            for i in range(math.floor(self._rangeStart/math.pi), math.ceil(self._rangeStop/math.pi)):
                for j in range(0, nSubticks):
                    phiTicks.append((i + j/nSubticks)*math.pi)

                    label = ""
                    if i < 0:
                        label += "-"

                    if j == 0:
                        label = r"$" + str(i) + r"\pi$"
                    else:
                        label = r"$\frac{" + str(nSubticks*i + j) + r"}{" + str(nSubticks) + r"}\pi$"

                    phiLabels.append(label)

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(7,8))
            fig.suptitle(self.getName(), fontsize=16)

            # Adjust subplot spacing:
            plt.subplots_adjust(left=0.14, right=0.86, bottom=0.1, top=0.9, wspace=0.2, hspace=0.6)

            ax1.set_title("Ball {rowaxis} Position".format(rowaxis=self._rowAxis))
            ax1.set_xlabel("φ [rad]")
            ax1.set_ylabel("{rowaxis} [px]".format(rowaxis=self._rowAxis))
            ax1.set_xticks(phiTicks)
            ax1.set_xticklabels(phiLabels)
            ax1.scatter(angles, xPositions, s=5, color=pointColor)
            if self.fitSuccessful():
                if self._xWave.isSecondOrderWave():
                    ax1.plot(self._fitFunctionAngles, fit2X, color=curveColor)
                else:
                    ax1.plot(self._fitFunctionAngles, fitX, color=curveColor)

                if self._xWave.matrixTransformFitSuccessful():
                    ax1.plot(self._fitFunctionAngles, fitMTx, color="red")
        
                ax1.plot(centerOfRotationX, centerOfRotation, color=resultColor, linestyle='--')


            ax2.set_title("Ball {colaxis} Position".format(colaxis=self._colAxis))
            ax2.set_xlabel("φ [rad]")
            ax2.set_ylabel("{colaxis} [px]".format(colaxis=self._colAxis))
            ax2.set_xticks(phiTicks)
            ax2.set_xticklabels(phiLabels)
            ax2.scatter(angles, yPositions, s=5, color=pointColor)
            if self.fitSuccessful():
                if self._yWave.isSecondOrderWave():
                    ax2.plot(self._fitFunctionAngles, fit2Y, color=curveColor)
                else:
                    ax2.plot(self._fitFunctionAngles, fitY, color=curveColor)

                #if self._yWave.matrixTransformFitSuccessful():
                #    ax2.plot(self._fitFunctionAngles, fitMTz, color="red")


            ax3.set_title("Trajectory")
            ax3.set_xlabel("{rowaxis} [px]".format(rowaxis=self._rowAxis))
            ax3.set_ylabel("{colaxis} [px]".format(colaxis=self._colAxis))

            if self.fitSuccessful():
                ax3.scatter(*self._axisPoints, color=resultColor, marker='x')
                #ax3.scatter(*self._apexPoints, color="green", marker='x')

            ax3.scatter(xPositions, yPositions, s=5, color=pointColor)

            if self.fitSuccessful():
                ax3.plot(fitX, fitY, color=curveColor)
                if self._xWave.isSecondOrderWave() and self._yWave.isSecondOrderWave():
                    ax3.plot(fit2X, fit2Y, color="green")

                ax3.plot(centerOfRotation, (self._y1intersect, self._y2intersect), color=resultColor)

            if (savePlot == True) and (saveAs != None):
                pictureFilename = saveAs
                plt.savefig(pictureFilename)

            if displayPlot == True:
                if interactive == True:
                    plt.ion()

                plt.show()

    def t(self, value, width=9, decimals=2):   # make into text
        text = ("{0:>"+str(width)+"}").format("-")
        if value != None:
            text = ("{0:"+str(width)+"."+str(decimals)+"f}").format(value)

        return text

    def getCoordinateList(self, withHeader=True):
        output = ""
        if withHeader==True:
            output = "  phi [rad]\tphi [deg]\t   {rowaxis} [px]\t   {colaxis} [px]\t{rowaxis}Fit [px]\t{colaxis}Fit [px]\t  {rowaxis}Fit2 [px]\t  {colaxis}Fit2 [px]\tradius [px]\n".format(rowaxis=self._rowAxis, colaxis=self._colAxis)

        for i in range(0, self.numberOfDatasets()):
            phi = self._xWave.getPhi(i, unit='rad')

            output += "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            self.t(self._xWave.getPhi(i, unit='rad'), decimals=4),
            self.t(self._xWave.getPhi(i, unit='deg'), decimals=2),
            self.t(self._xWave.getPoint(i)),
            self.t(self._yWave.getPoint(i)),
            self.t(self._xWave.getFitValue(phi=phi)),
            self.t(self._yWave.getFitValue(phi=phi)),
            self.t(self._xWave.getFitValue2(phi=phi)),
            self.t(self._yWave.getFitValue2(phi=phi)),
            self.t(self._rWave.getPoint(i)))

            output += "\n"

        return output


class ballSequence:
    """ Manages the ball image sequence to find the trajectory and fit the trajectory parameters. """

    def __init__(self, inputFileStack=None, outputFolder=None, darkFileStack=None, flatFileStack=None):
        self._initTime      = time.time()

        # BAMline CT:
        self._beamAxis = "z"
        self._rowAxis = "x"
        self._colAxis = "y"

        # HZG/DESY P05:
        """
        self._beamAxis = "Y"
        self._rowAxis = "X"
        self._colAxis = "Z"
        """

        self._inputFileStack  = None
        self._outputFolder    = None
        self._darkFileStack   = None
        self._flatFileStack   = None

        self.setInputFileStack(inputFileStack)
        self.setOutputFolder(outputFolder)
        self.setDarkFileStack(darkFileStack)
        self.setFlatFileStack(flatFileStack)

        self._scanAngle     = 360
        self._anglesFromHDF = False
        self._angles        = []    # List of angles for projection images.
        self._imageRotation = None
        self._imageFlipHorizontal = False
        self._imageFlipVertical   = False

        self._skip = 1   # only look at every n-th image file
        self._debugging = False
        self._displayMode = 'none'  # for matplotlib
        self._validDisplayModes = ('threshold', 'absorption', 'edges', 'croppedBall')
        self._saveIntermediates = False   # Save intermediate processing steps as pictures

        # List of ballData objects, accumulated during findBallTrajectory(),
        # but then converted into each trajectory's _xWave and _yWave before the sine fitting.
        self._ballList = []   

        self._multiprocessing = True
        self._nProcesses = 0

        self._applyDarks = False
        self._applyRefs  = False
        self._dark = None   # Mean dark image to apply during offset correction
        self._ref  = None   # Mean flat image to apply during flat field correction

        self._median = 1

        self._thresholdingRatio = 0.6   # set threshold to x% of min-max difference above minimum value
        self._thresholdAbsolute = None  # specify an absolute threshold value

        self._driftCompensation = True   # Try to compensate a drift of the background relative to the reference images.
        self._driftCompensationAmongRefs = False
        self._driftROI = False
        self._driftROItop    = None
        self._driftROIbottom = None
        self._driftROIleft   = None
        self._driftROIright  = None
        self._driftROIwidth  = None
        self._driftROIheight = None

        self._doCropping = False
        self._cropX0 = 0
        self._cropY0 = 0
        self._cropX1 = None
        self._cropY1 = None

        self._cropBorder = False
        self._cropTop    = None
        self._cropBottom = None
        self._cropLeft   = None
        self._cropRight  = None

        self._autocrop = True
        self._autocropSize = 1000
        self._autoCropBinningFactor = 40
        self._cropAndSaveCenteredBall = False
        self._centeredBallRadius = 400

        self._binning = 1

        self._edgeDetectionMode = 'sobel'
        self._validEdgeModes = ('sobel', 'canny')

        # Patch cleanup parameters: (when multiple patches are left over after thresholding)
        self._doPatchCleanup = True
        self._min_circle_area = 150*150      # Minimum number of pixels in the circle.
        self._max_patch_area = 1000*1000
        self._aspect_ratio_tolerance = 0.15  # Maximum aspect ratio deviation of a patch from 1 (=perfect circle aspect ratio)

        # Tolerances for circle fit:
        self._max_meanDeviation = 4    # maximum for the allowed mean deviation from circle fit [in pixels]
        self._max_maxDeviation  = 15   # maximum allowed deviation from circle fit for a single point [in pixels]

        # Intensity profile fit:
        self._doIntensityProfileFit = True
        self._intensityProfileFit_avgLines = 5

        # Sine fits:
        self._fitSecondOrder = True

        # For plotting:
        self._fig = None
        self._axes = None

        # Results:
        self._trajectory_circleFit    = None #ballTrajectory("Circle Fit", beamAxisName=self._beamAxis, rowAxisName=self._rowAxis, colAxisName=self._colAxis)
        self._trajectory_intensityFit = None #ballTrajectory("Intensity Profile Fit", beamAxisName=self._beamAxis, rowAxisName=self._rowAxis, colAxisName=self._colAxis)

    def log(self, message):
        if self._debugging == True:
            print(message)

    def setBeamAxis(self, name):
        self._beamAxis = name

    def setRowAxis(self, name):
        self._rowAxis = name

    def setColAxis(self, name):
        self._colAxis = name

    def setInputFileStack(self, inputFileStack):
        self._inputFileStack = createImageStack(inputFileStack)

    def setDarkFileStack(self, darkFileStack):
        self._darkFileStack = createImageStack(darkFileStack)
        if isinstance(self._darkFileStack, ImageStack):
            self._applyDarks = True
        else:
            self._applyDarks = False

    def setFlatFileStack(self, flatFileStack):
        self._flatFileStack = createImageStack(flatFileStack)
        if isinstance(self._flatFileStack, ImageStack):
            self._applyRefs = True
        else:
            self._applyRefs = False

    def setOutputFolder(self, outFolder=None):
        self._outputFolder = outFolder

    def getOutputFolder(self):
        return self._outputFolder

    def setScanAngle(self, scanAngle=360):
        self._scanAngle = scanAngle

    def rotate(self, rotation=None):
        if rotation == "0":
            rotation = None

        self._imageRotation = rotation
        
    def flip(self, horizontal=False, vertical=False):
        self._imageFlipHorizontal = horizontal
        self._imageFlipVertical   = vertical

    def getScanAngle(self, unit='deg'):
        return deg2x(self._scanAngle, unit)

    def getAngleForImage(self, projectionNumber):
        if projectionNumber < len(self._angles):
            return self._angles[projectionNumber]
        else:
            return None

    def skip(self, skip=1):
        if int(skip) == 0:
            skip = 1

        self._skip = int(skip)

    def showDebugInfo(self, debugging=False):
        self._debugging = debugging

    def displayMode(self, mode='threshold'):
        self._displayMode = mode

    def saveIntermediates(self, doSaveIntermediates=False):
        self._saveIntermediates = doSaveIntermediates

    def multiprocessing(self, mp=True):
        self._multiprocessing = mp

    def numberOfProcesses(self, n=1):
        self._nProcesses = n

    def applyDarks(self, doDarks=True):
        self._applyDarks = doDarks

    def applyRefs(self, doRefs=True):
        self._applyRefs = doRefs

    def median(self, median=1):
        self._median = median

    def edgeDetection(self, mode='sobel'):
        if mode in self._validEdgeModes:
            self._edgeDetectionMode = mode
        else:
            raise Exception("'{}' is not a valid edge detection mode.".format(mode))

    def threshold(self, ratio=0.6, absolute=None):
        if (ratio == None) or (absolute == None):
            self._thresholdingRatio = ratio
            self._thresholdAbsolute = absolute
        else:
            raise Exception("Thresholding: please set at least one of the parameters (ratio, absolute) to 'None' to choose the correct thresholding method.")

    def driftCompensation(self, refsToImg=True, amongRefs=False):
        self._driftCompensation = refsToImg
        self._driftCompensationAmongRefs = amongRefs

    def driftROI(self, top=None, bottom=None, left=None, right=None, width=500, height=500):
        self._driftROI       = True
        self._driftROItop    = top
        self._driftROIbottom = bottom
        self._driftROIleft   = left
        self._driftROIright  = right
        self._driftROIwidth  = width
        self._driftROIheight = height

    def getDriftROI(self, imgWidth=None, imgHeight=None):
        x0 = None
        y0 = None
        x1 = None
        y1 = None

        if self._driftROItop != None:
            y0 = self._driftROItop
            y1 = y0 + self._driftROIheight
        elif self._driftROIbottom != None:
            y1 = imgHeight - self._driftROIbottom
            y0 = y1 - self._driftROIheight

        if self._driftROIleft != None:
            x0 = self._driftROIleft
            x1 = x0 + self._driftROIwidth
        elif self._driftROIright != None:
            x1 = imgWidth - self._driftROIright
            x0 = x1 - self._driftROIwidth

        roi = ((x0, y0), (x1, y1))
        return roi

    def crop(self, x0, y0, x1, y1):
        self._doCropping = True
        self._cropX0 = x0
        self._cropY0 = y0
        self._cropX1 = x1
        self._cropY1 = y1

    def cropBorder(self, top=0, bottom=0, left=0, right=0):
        self._cropBorder = True
        self._cropTop    = top
        self._cropBottom = bottom
        self._cropLeft   = left
        self._cropRight  = right

    def autoCrop(self, doAutoCrop=True, autoCropSize=1000, autoCropBinningFactor=40):
        self._autocrop = doAutoCrop
        self._autocropSize = autoCropSize
        self._autoCropBinningFactor = autoCropBinningFactor

    def cropAndSaveCenteredBall(self, doCropBall=True, radius=400):
        self._cropAndSaveCenteredBall = doCropBall
        self._centeredBallRadius = radius

    def binning(self, binSize=1):
        self._binning = binSize

    def patchCleanup(self, doCleanUp=True, min_circle_area=(150*150), max_patch_area=(1000*1000), aspect_ratio_tolerance=0.15):
        self._doPatchCleanup         = doCleanUp
        self._min_circle_area        = min_circle_area
        self._max_patch_area         = max_patch_area
        self._aspect_ratio_tolerance = aspect_ratio_tolerance

    def circleFitTolerances(self, max_meanDeviation=4, max_maxDeviation=15):
        self._max_meanDeviation = max_meanDeviation
        self._max_maxDeviation  = max_maxDeviation

    def fitIntensityProfile(self, doFit=True):
        self._doIntensityProfileFit = doFit

    def getRuntime(self):
        return time.time() - self._initTime

    def getFormattedRuntime(self):
        rt = self.getRuntime()
        minutes = int(math.floor(rt / 60.0))
        seconds = rt - 60*minutes

        return "{} min {:.0f} s".format(minutes, seconds)

    def info(self):
        output = "Rotation Angles from HDF: {}\n".format(yesno(self._anglesFromHDF))
        if not self._anglesFromHDF:
            output += "Scan Angle: {} deg\n".format(self._scanAngle)

        output += "Row axis name:    {}\n".format(self._rowAxis)
        output += "Column axis name: {}\n".format(self._colAxis)
        output += "Beam axis name:   {}\n".format(self._beamAxis)
        
        output += "Image Rotation: {}\n".format(valOrZero(self._imageRotation))
        output += "Image Flip Horizontal: {}\n".format(yesno(self._imageFlipHorizontal))
        output += "Image Flip Vertical:   {}\n".format(yesno(self._imageFlipVertical))
        output += "Skip: {}\n".format(self._skip)
        output += "Apply Darks: {}\n".format(yesno(self._applyDarks))
        output += "Apply Refs:  {}\n".format(yesno(self._applyRefs))

        output += "Crop: "
        if (self._doCropping == True):
            output += "({}, {}) -- ({}, {})\n".format(self._cropX0, self._cropY0, self._cropX1, self._cropY1)
        elif (self._cropBorder == True):
            output += "border\n"
            output += "  Top:    {}\n".format(self._cropTop)
            output += "  Bottom: {}\n".format(self._cropBottom)
            output += "  Left:   {}\n".format(self._cropLeft)
            output += "  Right:  {}\n".format(self._cropRight)
        else:
            output += "no\n"

        output += "Auto Crop: {}\n".format(yesno(self._autocrop))

        output += "Drift Compensation:\n"
        output += "  Refs to Images: {}\n".format(yesno(self._driftCompensation))
        output += "  Among Refs:     {}\n".format(yesno(self._driftCompensationAmongRefs))

        output += "Drift ROI:\n"
        if self._driftROI == True:
            output += "  Top:    {}\n".format(self._driftROItop)
            output += "  Bottom: {}\n".format(self._driftROIbottom)
            output += "  Left:   {}\n".format(self._driftROIleft)
            output += "  Right:  {}\n".format(self._driftROIright)
            output += "  Width:  {}\n".format(self._driftROIwidth)
            output += "  Height: {}\n".format(self._driftROIheight)
        else:
            output += "no\n"

        output += "Binning: {}\n".format(self._binning)
        output += "Median: {}\n".format(self._median)

        output += "Threshold: "
        if self._thresholdingRatio != None:
            output += "ratio {:.3}\n".format(float(self._thresholdingRatio))
        elif self._thresholdAbsolute != None:
            output += "{}\n".format(self._thresholdAbsolute)
        else:
            output += "none\n"

        output += "Patch Cleanup: {}\n".format(yesno(self._doPatchCleanup))
        if self._doPatchCleanup:
            output += "  Min Patch Size: {}\n".format(self._min_circle_area)
            output += "  Max Patch Size: {}\n".format(self._max_patch_area)
            output += "  Aspect Ratio Tolerance: {:.3}\n".format(float(self._aspect_ratio_tolerance))

        output += "Edge Detection: {}\n".format(self._edgeDetectionMode)

        output += "Circle Fit Max(mean deviation): {}\n".format(self._max_meanDeviation)
        output += "Circle Fit Max(max deviation): {}\n".format(self._max_maxDeviation)

        output += "Fit Intensity Profile: {}\n".format(yesno(self._doIntensityProfileFit))
        if self._doIntensityProfileFit:
            output += "  Average Over Lines: {}\n".format(self._intensityProfileFit_avgLines)

        output += "Fit Second Order Waves: {}\n".format(yesno(self._fitSecondOrder))

        output += "Multiprocessing: {}\n".format(yesno(self._multiprocessing))
        if self._multiprocessing:
            if self._nProcesses > 0:
                output += "  Processes: {}\n".format(self._nProcesses)
            else:
                output += "  Processes: auto\n"

        output += "Time: {}\n".format(self.getFormattedRuntime())

        return output

    """
    def displayPictureFromList(self, i):
        if (self._displayMode in self._validDisplayModes):
            self.displayPicture(self._ballList[i], self._ballList[i].getDisplayX(), self._ballList[i].getDisplayY(), self._ballList[i].getDisplayRadius(), self._ballList[i].getX(), self._ballList[i].getY(), self._ballList[i].getFilename())

            # delete Pixmap to save RAM:
            self._ballList[i].setPixmap(0)
    """

    def displayPicture(self, img, cX=None, cY=None, radius=None, uncroppedX=None, uncroppedY=None, title=None, showROIs=True):
        if (self._displayMode in self._validDisplayModes):
            if (self._fig == None):  # New figure
                self._fig = plt.figure()
                self._axes = plt.subplot(1, 1, 1)
                plt.xticks(())
                plt.yticks(())

                # Adjust subplot spacing:
                plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0, hspace=0)

                plt.ion()
                plt.show()
                plt.pause(0.2)

            plt.figure(self._fig.number)
            plt.ion()
            self._axes.cla()
            self._axes.imshow(img.getPixelMap().astype('uint16'), cmap='gray')

            originalWidth  = img.getWidth()  * img.getResolution()
            originalHeight = img.getHeight() * img.getResolution()

            if (self._displayMode == "absorption") and showROIs:
                # Drift ROI rectangle:
                if self._driftROI and self._driftCompensation:
                    driftROI = self.getDriftROI(imgWidth=originalWidth, imgHeight=originalHeight)

                    x0 = driftROI[0][0] / img.getResolution()
                    y0 = driftROI[0][1] / img.getResolution()
                    x1 = driftROI[1][0] / img.getResolution()
                    y1 = driftROI[1][1] / img.getResolution()
                    width  = x1-x0
                    height = y1-y0

                    driftROI = patches.Rectangle((x0, y0), width, height, color='orange', fill=False, alpha=0.4)
                    self._axes.add_patch(driftROI)

                # Crop area rectangle:
                if self._doCropping or self._cropBorder:
                    x0 = self._cropX0/self._binning
                    y0 = self._cropY0/self._binning
                    width  = (self._cropX1 - self._cropX0) / self._binning
                    height = (self._cropY1 - self._cropY0) / self._binning

                    if x0 >= 0 and y0 >= 0 and width > 0 and height > 0:
                        cropArea = patches.Rectangle((x0, y0), width, height, color='blue', fill=False, alpha=0.4)
                        self._axes.add_patch(cropArea)

            if uncroppedX != None and uncroppedY != None:
                subtitle = "Ball at ({rowaxis}={x:.2f}, {colaxis}={y:.2f})".format(rowaxis=self._rowAxis, x=uncroppedX, colaxis=self._colAxis, y=uncroppedY)
                self._axes.set_title(subtitle)

            if title != None:
                self._fig.suptitle(title, fontsize=16)
                #self._axes.set_title(title)

            # Circle:
            if (cX != None) and (cY != None) and (radius != None):
                circle = patches.Circle((cX, cY), radius, color='red', linewidth=2, fill=False, alpha=0.5)
                self._axes.add_patch(circle)

            self._axes.relim()
            self._axes.autoscale_view()

            self._fig.canvas.flush_events()
            time.sleep(0.1)
            self._fig.canvas.flush_events()
            time.sleep(0.1)

    def savePicture(self, ballImg, ballData, subfolder):
        folder = self.getOutputFolder() + "/" + subfolder
        if not os.path.exists(folder):
            os.makedirs(folder)
        ballImg.save(folder + "/" + ballData.getFilename())

    def trackBall(self):
        self._ballList = []

        # Results:
        self._trajectory_circleFit    = ballTrajectory(name="Circle Fit", beamAxisName=self._beamAxis, rowAxisName=self._rowAxis, colAxisName=self._colAxis)
        self._trajectory_intensityFit = ballTrajectory(name="Intensity Profile Fit", beamAxisName=self._beamAxis, rowAxisName=self._rowAxis, colAxisName=self._colAxis)


        if isinstance(self._inputFileStack, ImageStack): 
            self._inputFileStack.buildStack()
        else:
            raise Exception("No input files given.")


        """
        # Try to find HDF info file to get angles:
        for f in fileList:
            file = self.getInputFolder() + "/" + f
            if os.path.isfile(file):
                if f.endswith(".h5"):
                    h5file = h5py.File(file, mode='r')
                    self._angles = h5file["entry/scan/data/s_rot/value"][()]
                    h5file.close()

                    n_angles = len(self._angles)
                    self._anglesFromHDF = True

                    print("Found {} rotation angles in HDF file:".format(n_angles))
                    print(f + "\n")
        """

        # Iterate through projection images:

        processedImageCounter = 0   # number of processed (unskipped) images
        projectionCounter = 0       # total number of projection files (img, no darks/refs)
        ballDataList = []

        if self._inputFileStack.nSlices() > 0:
            # Prepare dark & flat field correction:
            if self._applyDarks:
                if isinstance(self._darkFileStack, ImageStack):
                    self._darkFileStack.buildStack()
                    if self._darkFileStack.nSlices() > 0:
                        self._dark = self._darkFileStack.getMeanImage()
                        self._dark.rotate(rotation=self._imageRotation)
                        self._dark.flip(horizontal=self._imageFlipHorizontal, vertical=self._imageFlipVertical)
                    else:
                        self.log("No dark field file(s) found that match the given name pattern.")
                else:
                    self.log("No dark field image stack given.")

            if self._applyRefs:
                if isinstance(self._flatFileStack, ImageStack):
                    self._flatFileStack.buildStack()
                    if self._flatFileStack.nSlices() > 0:
                        self._ref = self._flatFileStack.getMeanImage()
                        self._ref.rotate(rotation=self._imageRotation)
                        self._ref.flip(horizontal=self._imageFlipHorizontal, vertical=self._imageFlipVertical)

                        if self._applyDarks:
                            self._ref.applyDark(self._dark)
                    else:
                        self.log("No flat field file(s) found that match the given name pattern.")
                else:
                    self.log("No flat field image stack given.")

            for i in range(self._inputFileStack.nSlices()):
                progress = 100*(float(i+1)/float(self._inputFileStack.nSlices()))
                print("Image {}/{} ({:0.1f}%)  ".format((i+1), self._inputFileStack.nSlices(), progress), end='\r')

                projectionCounter += 1

                # Skipping every n-th image:
                if math.fmod(int(projectionCounter), int(self._skip)) != 0:
                    continue

                ballData = ballDataset(filename=self._inputFileStack.getFileBasename(index=i), index=i)

                if self._multiprocessing == True:
                    ballDataList.append(ballData)
                else:
                    self._ballList.append(self.processProjectionImage(ballData))

                processedImageCounter += 1

        else:
            self.log("No input projection file(s) found that match the given name pattern.")

        # Multiprocessing:
        # Give all images in the list to the processing pool:
        self.processImagesInBuffer(ballDataList)
        ballDataList = []    # clear image list

        # Once we know how many images were counted,
        # and once the list is filled, we can assign an angle to each point,
        # if no HDF5 info file had been encountered.
        if len(self._ballList) == processedImageCounter:
            for i in range(processedImageCounter):
                angle = i * self._skip * (self.getScanAngle(unit='deg') / projectionCounter)
                self._ballList[i].setAngle(angle, unit='deg')
        else:
            raise Exception("Number of processed images ({nImg}) doesn't match number of processed datasets ({nDatasets}).".format(nImg=processedImageCounter, nDatasets=len(self._ballList)))


        # Convert self._ballList (list of ballData objects) into trajectories:
        self._trajectory_circleFit.clear()
        self._trajectory_intensityFit.clear()

        for point in self._ballList:
            # trajectory.add() automatically rejects misfits (i.e. invalid points with 'None' coordinates)
            if point.isAccepted():
                self._trajectory_circleFit.add(point.getAngle(unit='rad'), point.getX(), point.getY(), point.getRadius())

        if self._doIntensityProfileFit:
            for point in self._ballList:
                if point.isAccepted():
                    self._trajectory_intensityFit.add(point.getAngle(unit='rad'), point.getXint(), point.getYint(), point.getRadiusInt())


    def processImagesInBuffer(self, ballDataList):  # Multiprocessing
        if self._multiprocessing == True:
            #plt.ioff()
            plt.close('all')
            self._fig  = None
            self._axes = None

            if len(ballDataList) > 0:
                if self._nProcesses > 0:
                    pool = Pool(processes=self._nProcesses)
                else:
                    pool = Pool()   # let Python handle the number of processes

                #ballList = pool.map_async(self.processProjectionImage, ballDataList, chunksize=1, callback=self.processFinished)
                #pool.close()
                #pool.join()
                #self._ballList.extend(ballList)

                for ballData in pool.imap(self.processProjectionImage, ballDataList, chunksize=1):
                    self._ballList.append(ballData)

    def processProjectionImage(self, ballData):
        projectionIndex = ballData.getProjectionIndex()
        ballImg = self._inputFileStack.getImage(index=projectionIndex)
        ballImg.rotate(rotation=self._imageRotation)
        ballImg.flip(horizontal=self._imageFlipHorizontal, vertical=self._imageFlipVertical)

        displayImage = None  # any picture to be displayed with matplotlib

        # Centered ball images:
        pxCB_absorption = None
        pxCB_median     = None
        pxCB_threshold  = None
        pxCB_patches    = None
        pxCB_edges      = None

        # Apply the dark and reference image to get absorption picture:
        if self._applyDarks:
            if(self._dark.isSet()):
                ballImg.applyDark(self._dark)

        if self._applyRefs:
            if(self._ref.isSet()):
                if self._driftCompensation:
                    shift = (0, 0)

                    if self._driftROI == False:
                        shift = ballImg.calcRelativeShift(self._ref)
                    else:
                        # Crop images to drift ROI
                        driftROI = self.getDriftROI(imgWidth=ballImg.getWidth(), imgHeight=ballImg.getHeight())
                        x0 = driftROI[0][0]
                        y0 = driftROI[0][1]
                        x1 = driftROI[1][0]
                        y1 = driftROI[1][1]

                        croppedRef = copy.deepcopy(self._ref)
                        croppedRef.crop(x0, y0, x1, y1)

                        croppedImg = copy.deepcopy(ballImg)
                        croppedImg.crop(x0, y0, x1, y1)

                        shift = croppedImg.calcRelativeShift(croppedRef)

                    self.log("Shift: {}".format(shift))

                    shiftedRef = copy.deepcopy(self._ref)
                    shiftedRef.setPixelMap(self._ref.getShiftedPixmap(shift[1], shift[0]))
                    ballImg.applyFlatfield(shiftedRef, rescaleFactor=1)
                    ballImg.clip(0, 1)
                    ballImg.renormalize(newMin=0, newMax=60000, currentMax=1)
                else:
                    ballImg.applyFlatfield(self._ref, rescaleFactor=1)
                    ballImg.clip(0, 1)
                    ballImg.renormalize(newMin=0, newMax=60000, currentMax=1)

        originalWidth = ballImg.getWidth()
        originalHeight= ballImg.getHeight()

        # Save absorption picture for 2D fit of intensity function and autoCrop.
        originalAbsorptionImage = copy.deepcopy(ballImg)

        # Binning:
        if self._binning > 1:
            ballImg.bin(binSizeX=self._binning, binSizeY=self._binning, operation="mean")

        if self._saveIntermediates:
            self.savePicture(ballImg, ballData, "01_Absorption_Binned")

        # Save absorption picture for Matplotlib:
        if self._displayMode == 'absorption':
            displayImage = copy.deepcopy(ballImg)

        # Cropping
        if self._doCropping:
            ballImg.crop(self._cropX0/self._binning, self._cropY0/self._binning, self._cropX1/self._binning, self._cropY1/self._binning)
        elif self._cropBorder:
            self._cropX0 = self._cropLeft
            self._cropX1 = originalWidth - self._cropRight
            self._cropY0 = self._cropTop
            self._cropY1 = originalHeight - self._cropBottom

            ballImg.cropBorder(self._cropTop/self._binning, self._cropBottom/self._binning, self._cropLeft/self._binning, self._cropRight/self._binning)

        if self._autocrop:
            preview = copy.deepcopy(originalAbsorptionImage)
            preview.bin(binSizeX=self._autoCropBinningFactor, binSizeY=self._autoCropBinningFactor, operation="mean")
            minIndex = numpy.argmin(preview.getPixelMap(), axis=None)
            minimum  = numpy.unravel_index(minIndex, preview.getPixelMap().shape)
            minX = (minimum[1] * self._autoCropBinningFactor - self._cropX0)/self._binning
            minY = (minimum[0] * self._autoCropBinningFactor - self._cropY0)/self._binning
            roiSize = self._autocropSize/self._binning

            ballImg.cropROIaroundPoint(minX, minY, roiSize, roiSize)

        if self._doCropping or self._cropBorder or self._autocrop:
            if self._saveIntermediates:
                self.savePicture(ballImg, ballData, "02_Cropped")

        # Save absorption image if centered ball crop should be saved afterwards:
        if self._cropAndSaveCenteredBall:
            pxCB_absorption = copy.deepcopy(ballImg)


        # Median filter:
        if self._median > 1:
            ballImg.applyMedian(kernelSize=self._median)
            if self._saveIntermediates:
                self.savePicture(ballImg, ballData, "03_Median")

        # Save median image if centered ball crop should be saved afterwards:
        if self._cropAndSaveCenteredBall:
            pxCB_median = copy.deepcopy(ballImg)

        # Thresholding:
        if self._thresholdingRatio != None:
            threshold = int(ballImg.getMin() + self._thresholdingRatio * (ballImg.getMax() - ballImg.getMin()))
            ballImg.applyThreshold(threshold)
        elif self._thresholdAbsolute != None:
            ballImg.applyThreshold(self._thresholdAbsolute)

        if self._saveIntermediates:
            self.savePicture(ballImg, ballData, "04_Thresholding")

        if self._displayMode == 'threshold':
            displayImage = copy.deepcopy(ballImg)   # to display with matplotlib

        # Save threshold image if centered ball crop should be saved afterwards:
        if self._cropAndSaveCenteredBall:
            pxCB_threshold = copy.deepcopy(ballImg)


        ballImg.invert()

        # Label patches and remove invalid ones (too big, wrong aspect ratios...)
        nPatches = 0
        nRemoved = 0
        nRemaining = 1
        if self._doPatchCleanup:
            nPatches, nRemoved, nRemaining, patchGeometry = ballImg.cleanPatches(min_patch_area=self._min_circle_area, max_patch_area=self._max_patch_area, remove_border_patches=True, aspect_ratio_tolerance=self._aspect_ratio_tolerance)
            
            if self._saveIntermediates:
                self.savePicture(ballImg, ballData, "05_Patches_removed")

            # Save patched image if centered ball crop should be saved afterwards:
            if self._cropAndSaveCenteredBall:
                pxCB_patches = copy.deepcopy(ballImg)

        displayCircleCenterX = 0
        displayCircleCenterY = 0
        displayCircleRadius  = 0

        if nRemaining == 1:  # Only accept images with exactly one acceptable patch
            # Edge detection:
            ballImg.filter_edges(mode=self._edgeDetectionMode)

            # Binarize:
            thresh = (ballImg.max()+ballImg.min())/2.0
            ballImg.applyThreshold(threshold=thresh, lower=0, upper=60000)

            if self._displayMode == 'edges':
                displayImage = copy.deepcopy(ballImg)   # to display with matplotlib


            if self._saveIntermediates:
                self.savePicture(ballImg, ballData, "06_Edges")

            # Save edge image if centered ball crop should be saved afterwards:
            if self._cropAndSaveCenteredBall:
                pxCB_edges = copy.deepcopy(ballImg)


            # Least squares circle fit:
            centerX, centerY, radius, meanDifference, minDifference, maxDifference = ballImg.fitCircle()


            # Save cropped balls:
            if self._cropAndSaveCenteredBall:
                try:
                    cropX = round(centerX)
                    cropY = round(centerY)
                    cropROIsize = 2 * self._centeredBallRadius / self._binning

                    if pxCB_absorption != None:
                        pxCB_absorption.cropROIaroundPoint(cropX, cropY, cropROIsize, cropROIsize)
                        self.savePicture(pxCB_absorption, ballData, "01_Absorption_Binned/centered")

                    if pxCB_median != None:
                        pxCB_median.cropROIaroundPoint(cropX, cropY, cropROIsize, cropROIsize)
                        self.savePicture(pxCB_median, ballData, "03_Median/centered")

                    if pxCB_threshold != None:
                        pxCB_threshold.cropROIaroundPoint(cropX, cropY, cropROIsize, cropROIsize)
                        self.savePicture(pxCB_threshold, ballData, "04_Thresholding/centered")

                    if pxCB_patches != None:
                        pxCB_patches.cropROIaroundPoint(cropX, cropY, cropROIsize, cropROIsize)
                        self.savePicture(pxCB_patches, ballData, "05_Patches_removed/centered")

                    if pxCB_edges != None:
                        pxCB_edges.cropROIaroundPoint(cropX, cropY, cropROIsize, cropROIsize)
                        self.savePicture(pxCB_edges, ballData, "06_Edges/centered")
                except Exception as e:
                    log("Warning: something went wrong cropping the centered ball. Maybe your crop ROI is out of bounds with the image (e.g. crop radius too big?) {}".format(e))


            # Rescale mean and max deviation to unbinned image:
            meanDifference *= ballImg.getResolution()
            maxDifference  *= ballImg.getResolution()

            if self._displayMode == 'threshold' or self._displayMode == 'edges':
                displayCircleCenterX = centerX
                displayCircleCenterY = centerY
                displayCircleRadius  = radius

            # Shift circle parameters to uncropped coordinate system of (binned) picture:
            centerX += ballImg.getBoundingBoxX0()
            centerY += ballImg.getBoundingBoxY0()

            if self._displayMode == 'absorption':
                displayCircleCenterX = centerX
                displayCircleCenterY = centerY
                displayCircleRadius  = radius

            # Shift circle parameters to unbinned coordinate system of original picture:
            centerX *= float(ballImg.getResolution())
            centerY *= float(ballImg.getResolution())
            radius  *= float(ballImg.getResolution())

            # Add this to the list of data points if it
            # is in agreement with the fit tolerance parameters:
            if (meanDifference < self._max_meanDeviation) and (maxDifference < self._max_maxDeviation):
                ballData.setX(centerX)
                ballData.setY(centerY)
                ballData.setRadius(radius)
                ballData.setMeanDev(meanDifference)
                ballData.setMaxDev(maxDifference)
                ballData.setAccepted(True)

                # 2D intensity fit:
                if self._doIntensityProfileFit:

                    intensityMap = copy.deepcopy(originalAbsorptionImage)
                    intensityMap.cropROIaroundPoint(centerX, centerY, 1.95*radius, 1.95*radius)
                    intensityMap.applyMedian(kernelSize=5)

                    if self._saveIntermediates:
                        self.savePicture(intensityMap, ballData, "07_Ball_IntensityFit")

                    # Intensity profile fit along central horizontal line:
                    fittedI0x, fittedMux, fittedRx, fittedX0 = intensityMap.fitIntensityProfile(axis='x')

                    # Intensity profile fit along central vertical line:
                    fittedI0y, fittedMuy, fittedRy, fittedY0 = intensityMap.fitIntensityProfile(axis='y')

                    fittedR = None
                    if (fittedRx != None) and (fittedRy != None):
                        fittedR = (fittedRx + fittedRy) / 2.0
                    elif (fittedRy == None):
                        fittedR = fittedRx
                    elif (fittedRx == None):
                        fittedR = fittedRy

                    if (fittedX0 != None):
                        fittedX0 += intensityMap.getBoundingBoxX0()
                     
                    if (fittedY0 != None):
                        fittedY0 += intensityMap.getBoundingBoxY0()

                    ballData.setXint(fittedX0)
                    ballData.setYint(fittedY0)
                    ballData.setRadiusInt(fittedR)

            print(self.getCircleFitInfo(ballData, nRemaining, nPatches, meanDifference, maxDifference))

        else:   # Not exactly 1 patch meets the circle criteria. Display but don't fit.
            print(self.getCircleFitInfo(ballData, nRemaining, nPatches))

        if (self._displayMode in self._validDisplayModes):

            ballData.setDisplayX(displayCircleCenterX)
            ballData.setDisplayY(displayCircleCenterY)
            ballData.setDisplayRadius(displayCircleRadius)

            if (self._multiprocessing == False):
                if displayImage != None:
                    self.displayPicture(displayImage, displayCircleCenterX, displayCircleCenterY, displayCircleRadius, centerX, centerY, ballData.getFilename())

        return ballData

    def getCircleFitInfo(self, ballData, nRemaining, nPatches, meanDev=None, maxDev=None):
        if nRemaining > nPatches:
            nRemaining = 0

        angleString = ""
        if ballData.getAngle(unit='deg') != None:
            angleString = "{}".format("{:.2f}".format(ballData.getAngle(unit='deg')).rjust(7))

        patchesString = "{}/{} patches,".format(str(nRemaining).rjust(3), str(nPatches).ljust(3))
        
        deviationString = ""
        if (meanDev != None) and (maxDev != None):
            deviationString = "mean/max: {}  {} " .format("{:.3f}".format(meanDev).rjust(7), "{:.3f}".format(maxDev).rjust(7))

        accepted = "X"
        if ballData.isAccepted():
            accepted = "OK"

        output = "{} {}  {}  {}{}".format(angleString, ballData.getFilename(), patchesString, deviationString, accepted)

        return output

    def calcAxisParameters(self, fitSecondOrderWave=True):
        self._fitSecondOrder = fitSecondOrderWave
        print("\n")

        print("Calculating Axis Parameters. Number of datasets: {}".format(self._trajectory_circleFit.numberOfDatasets()))
        if self._trajectory_circleFit.numberOfDatasets() > 0:
            print("Analyzing Trajectory...")
            self._trajectory_circleFit.analyseTrajectory(self._fitSecondOrder)

        if self._trajectory_intensityFit.numberOfDatasets() <= 0:
            self._doIntensityProfileFit = False

        if self._doIntensityProfileFit:
            for point in self._ballList:
                self._trajectory_intensityFit.add(point.getXint(), point.getYint(), point.getAngle(unit='rad'))

            self._trajectory_intensityFit.analyseTrajectory(self._fitSecondOrder)

    def plotTrajectories(self, displayPlot=True, savePlot=True, saveAs=None, infosInFilename=True):
        if (savePlot == True) or (displayPlot == True):
            filename_circleFit    = None
            filename_intensityFit = None

            if savePlot == True:
                basename = ""
                if (saveAs != None):
                    basename = saveAs

                filename_circleFit = self.prepareFilename(basename=basename, specifier="graph_circleFit", extension="png", addInfos=infosInFilename)

                filename_intensityFit = self.prepareFilename(basename=basename, specifier="graph_intensityFit", extension="png", addInfos=infosInFilename)

            # Turn on interactive mode for first plot, to allow second plot popping up simultaneously.
            # Otherwise, first plot needs to be non-interactive to persist.
            ia = False
            if self._doIntensityProfileFit:
                ia = True

            if self._trajectory_circleFit.numberOfDatasets() > 0:
                self._trajectory_circleFit.plotResults(displayPlot, savePlot, filename_circleFit, interactive=ia)

            if self._doIntensityProfileFit and (self._trajectory_intensityFit.numberOfDatasets() > 0):
                self._trajectory_intensityFit.plotResults(displayPlot, savePlot, filename_intensityFit, interactive=False, curveColor='darkgray')

    def prepareFilename(self, basename, specifier=None, extension="txt", addInfos=True):
        filename = self.getOutputFolder() + "/" + basename
        if addInfos:
            filename += "_med{}_bin{}_skip{}".format(self._median, self._binning, self._skip)

        if specifier != None:
            filename += "_" + specifier

        filename += "." + extension

        if not os.path.exists(self.getOutputFolder()):
            os.makedirs(self.getOutputFolder())

        return filename

    def saveCoordinates(self, name="ballFit", specifier="coordinates", infosInFilename=True, withHeader=True):
        if name != None:
            if (self._trajectory_circleFit.numberOfDatasets() > 0):
                circleFitFilename = self.prepareFilename(basename=name, specifier=specifier+"_circleFit", extension="txt", addInfos=infosInFilename)
                circleFitOutput = self._trajectory_circleFit.getCoordinateList(withHeader)

                f = open(circleFitFilename, 'w')
                f.write(circleFitOutput)
                f.close()

            if self._doIntensityProfileFit and (self._trajectory_intensityFit.numberOfDatasets() > 0):
                intensityFitFilename = self.prepareFilename(basename=name, specifier=specifier+"_intensityFit", extension="txt", addInfos=infosInFilename)
                intensityFitOutput = self._trajectory_intensityFit.getCoordinateList(withHeader)

                g = open(intensityFitFilename, 'w')
                g.write(intensityFitOutput)
                g.close()

    def importCoordinates(self, name="ballFit", specifier="coordinates_circleFit", infosInFilename=True, filename=None):
        if (name != None) or (filename != None):
            if filename != None:
                coordinatesFile = filename
            else:
                coordinatesFile = self.prepareFilename(basename=name, specifier=specifier, extension="txt", addInfos=infosInFilename)

            if os.path.isfile(coordinatesFile) == True:
                if "intensityFit" in coordinatesFile:
                    self._trajectory_intensityFit.clear()
                else:
                    self._trajectory_circleFit.clear()

                with open(coordinatesFile) as f:
                    for line in f:
                        # Ignore lines with any letters except e and E (e.g. header line):
                        if re.search('[a-df-zA-DF-Z]', line):
                            continue

                        values = re.split(r'\t+', line)
                        phi = float(values[0].strip())
                        x   = float(values[2].strip())
                        y   = float(values[3].strip())
                        r   = float(values[8].strip())

                        if "intensityFit" in coordinatesFile:
                            self._trajectory_intensityFit.add(phi, x, y, r)
                        else:
                            self._trajectory_circleFit.add(phi, x, y, r)
            else:
                raise Exception("Cannot find coordinate file in:\n{}\nYou can use the 'filename' parameter to specify a coordinate file.".format(coordinatesFile))


    def getAllGeometryResults(self, extended=False):
        output  = self._trajectory_circleFit.formatGeometryResults(extended)
        #output += self._trajectory_circleFit.formatMatrixTransformationFitResults()

        if self._doIntensityProfileFit:
            output += "\n\n"
            output += self._trajectory_intensityFit.formatGeometryResults(extended)
            #output += self._trajectory_intensityFit.formatMatrixTransformationFitResults()

        return output

    def saveParameters(self, name="ballFit", specifier="parameters", infosInFilename=True):
        if name != None:
            filename = self.prepareFilename(basename=name, specifier=specifier, extension="txt", addInfos=infosInFilename)
            
            fileOutput  = "Analysis parameters:\n"
            fileOutput += "---------------------------------\n"
            fileOutput += self.info()

            f = open(filename, 'w')
            f.write(fileOutput)
            f.close()

    def saveGeometryResults(self, name="ballFit", specifier="results", infosInFilename=True):
        if name != None:
            filename = self.prepareFilename(basename=name, specifier=specifier, extension="txt", addInfos=infosInFilename)
            
            fileOutput = self.getAllGeometryResults(extended=True)

            f = open(filename, 'w')
            f.write(fileOutput)
            f.close()

            print("\nSaved results to:\n{}\n".format(filename))
