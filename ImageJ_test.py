"""
from PyQt5 import QtCore, QtGui, QtWidgets
import qimage2ndarray
from PyQt5.uic import loadUiType
import numpy
from PIL import Image
import os
import time
import tkinter.filedialog
from PyQt5.QtGui import QIcon, QPixmap
from scipy import ndimage

import sys
import jnius
import imagej
import scyjava

scyjava.config.add_option('-Xmx6g')
ij = imagej.init('C:/Users/hmarkoet/OneDrive_BAM/OneDrive - Bundesanstalt für Materialforschung und -prüfung (BAM)/Dokumente/Fiji.app')


path_klick = tkinter.filedialog.askopenfilename(title="Select one file of the scan")
im_000deg = Image.open(path_klick)

#jimage = ij.io().open(path_klick)
ij.py.show(im_000deg, cmap='gray')

#import numpy
import time
import socket
import struct
import numpy as np
from skimage.color import rgb2gray


def send_to_imagej(img, address, title='Image'):

    max_ = img.max()

    if img.dtype.kind == 'f':
        if max_ <= 1:
            img *= 2**16 - 1
            max_ *= 2**16 - 1

    if max_ <= 255:
        img = img.astype(np.int16)
        img -= 127
        img = img.astype(np.int8)
        dtype = b'b'

    else:
        # check for 12 bit - looking images (or just improve contrast)
        if max_ <= 2**12:
            img *= 2**4
        img = img.astype(np.int32)
        img -= 2 ** 15 - 1
        img = img.astype(np.int16)
        img += np.random.randint(-1000, 1000)
        dtype = b'h'

    img = rgb2gray(img)

    data = struct.pack('>ccHH100s',
                       b'1', dtype, img.shape[0], img.shape[1],
                        title.encode('utf-8'))
    send_over_socket(address, data + img.tobytes())


def send_over_socket(address, data):
    if isinstance(address, (tuple, list)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    else:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    try:
        sock.connect(address)
    except socket.error:
        time.sleep(0.1)
        try:
            sock.connect(address)
        except socket.error as e:
            raise socket.error('Could not open "%s": %s' % (address, e))

    try:
        sock.sendall(data)
    finally:
        sock.close()


if __name__ == '__main__':
    from skimage import data, img_as_float, img_as_uint
    moon = np.zeros((600,400))

    i = 0
    while 1:
        i += 1
        send_to_imagej(moon, ('', 5048), 'My Very Very Special Image %s' % i)
        time.sleep(0.1)
"""


"""
# Live streaming 2D monochrome image receiver

# We can start this from python to view images
# So we get the live image or a current image from Python, and
# send it to ImageJ through this interface.  Boom!

from ij import IJ
from ij.process import ByteProcessor, ShortProcessor, LUT

import jarray

import socket
import struct
import sys


if len(sys.argv) > 1:
    PORT = sys.argv[1]
else:
    PORT = 5048

# create a LUT that highlights 0-1 as blue and 254-255 as red
red = jarray.zeros(256, 'b')
for i in range(254):
    red[i] = (i - 128)

green = jarray.zeros(256, 'b')
for i in range(2, 254):
    green[i] = (i - 128)

blue = jarray.zeros(256, 'b')
for i in range(2, 256):
    blue[i] = (i - 128)

cm = LUT(blue, green, red)

# create a random image and display it
from java.util import Random
imp = IJ.createImage("A Random Image", "8-bit", 512, 512, 1)
Random().nextBytes(imp.getProcessor().getPixels())
imp.getProcessor().setLut(cm)
imp.show()

# start a socket to listen for image data
serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serversock.bind(('', PORT))
serversock.listen(1)

while 1:
    clientsock, addr = serversock.accept()

    # get the header info
    fmt = 'ccHH100s'
    header = clientsock.recv(struct.calcsize(fmt))
    reserved, dtype, width, height, title = struct.unpack(fmt, header)

    # get all data until the socket closes
    try:
        buf = []
        while True:
            data = clientsock.recv(4096)
            if data:
                buf.append(data)
            else:
                break
        buf = ''.join(buf)
    finally:
        clientsock.close()

    arr = struct.unpack('<%s%s'% (width * height, dtype), buf)
    jarr = jarray.array(arr, dtype)

    if dtype == 'h':
        ip = ShortProcessor(width, height, jarr, cm)
    else:
        ip = ByteProcessor(width, height, jarr, cm)

    imp.setProcessor(ip)
    imp.setTitle(title)
    imp.updateAndDraw()
"""

# Create an ImageJ gateway with the newest available version of ImageJ.
import imagej as ij
ij = ij.init(headless=False)
#ij = ij.init('C:/Users/hmarkoet/OneDrive_BAM/OneDrive - Bundesanstalt für Materialforschung und -prüfung (BAM)/Dokumente/Fiji.app')

# Load an image.
image_url = 'https://samples.fiji.sc/new-lenna.jpg'
jimage = ij.io().open(image_url)

# Convert the image from ImageJ to xarray, a package that adds
# labeled datasets to numpy (http://xarray.pydata.org/en/stable/).
image = ij.py.from_java(jimage)

# Display the image (backed by matplotlib).
ij.py.show(image, cmap='gray')
