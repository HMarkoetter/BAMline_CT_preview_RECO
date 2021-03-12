

import numpy as np
import h5py

hdf = h5py.File('C:\HDF5-Reading-Writing\dark__210303_Zahn13_45_width_Rocking_extended_7112eV_001.h5', 'r')
ls = list(hdf.keys())
print('List of datasets in this file: \n', ls)
entry = hdf.get('entry')
print('Items in entry:', list(entry.items()))
data = entry.get('/entry/data')
#dataset1 = np.array(data)
#print('Shape of dataset: \n', dataset1.shape)
k = list(data.attrs.keys())
v = list(data.attrs.values())
print(k)
print(v)
print(data.attrs[k[0]])
hdf.close()