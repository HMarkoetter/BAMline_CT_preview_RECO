
import urllib.request
import numpy as np
import ssl
import matplotlib.pyplot as plt

import shutil

from PIL import Image
import PIL
import io

#The url to our security camera
#this url sends stream of byte data
#this data contains frames in jpeg format
#jpge starts with : b'\xff\xd8'
#and ends on : b'\xff\xd9'
#thats why we need to find the jpeg data in between these byte data

url = 'http://cam-window/mjpg/video.mjpg?'

#opens the stream object
stream = urllib.request.urlopen(url)
#creates a variable as byte
bytes = b''# MAKE IT BYTES

#we need to start a loop so that we get the data coming from the stream object
jpg =0
while jpg == 0: # this will stop when we have the first meaningfull frame from the stream of the data

    #we read the stream ever 1024th character
    bytes += stream.read(1024)
    #find the jpeg
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    #if any jpeg found :
    if a != -1 and b != -1:
        print('I have got the jpeg')

        jpg = bytes[a:b+2]


        image = Image.open(io.BytesIO(jpg))

        image.show()
        #can also save
        #image.save("geeks.jpg")


