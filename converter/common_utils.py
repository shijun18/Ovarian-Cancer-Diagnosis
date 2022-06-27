import os
import h5py
import numpy as np
import SimpleITK as sitk
import glob
import pydicom

def nii_reader(data_path):
    data = sitk.ReadImage(data_path)
    image = sitk.GetArrayFromImage(data).astype(np.float32)
    return data,image


def save_as_hdf5(data,save_path,key=None):
    '''
    Numpy array save as hdf5.

    Args:
    - data: numpy array
    - save_path: string, destination path
    - key: string, key value for reading
    '''
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()


def hdf5_reader(data_path,key=None):
    '''
    Hdf5 file reader, return numpy array.
    '''
    hdf5_file = h5py.File(data_path,'r')
    image = np.asarray(hdf5_file[key],dtype=np.float32)
    hdf5_file.close()

    return image


def dicom_series_reader(data_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(data_path)
    reader.SetFileNames(dicom_names)
    data = reader.Execute()
    image_array = sitk.GetArrayFromImage(data).astype(np.float32)

    return data,image_array  


