# -*- coding: UTF-8 -*-

import math

def log(message):
    """Print an output message."""
    print(message)

def yesno(b):
	if b == True:
		return "yes"
	else:
		return "no"

def valOrZero(v):
	if v == None:
		return "0"
	else:
		return v

def rad2deg(rad):
	return rad * 180.0 / math.pi

def deg2rad(deg):
	return deg * math.pi / 180.0

# Convert to unit, assuming angle is in deg:
def deg2x(angle, unit='deg'): 
	if unit == 'deg':
		return angle
	elif unit == 'rad':
		if angle != None:
			return deg2rad(angle)
		else:
			return None
	else:
		raise Exception("Angle unit must be 'deg' or 'rad'.")

# Convert to unit, assuming angle is in rad:
def rad2x(angle, unit='rad'):
	if unit == 'rad':
		return angle
	elif unit == 'deg':
		if angle != None:
			return rad2deg(angle)
		else:
			return None
	else:
		raise Exception("Angle unit must be 'deg' or 'rad'.")