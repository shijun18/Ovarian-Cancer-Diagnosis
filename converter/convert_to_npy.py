import os
import numpy as np
import shutil
from common_utils import save_as_hdf5, nii_reader,dicom_series_reader



def convert_to_npy(input_path,save_path):
    '''
    Convert the raw data(e.g. dcm and *.nii) to numpy array and save as hdf5.
    '''
    ID = []
    
    for subdir in os.scandir(input_path):
        sub_save_path = os.path.join(save_path,subdir.name)
        if os.path.exists(sub_save_path):
            shutil.rmtree(sub_save_path)
        os.makedirs(sub_save_path)

        for item in os.scandir(subdir.path):
            if item.is_dir():
                ID = item.name.replace(' ','')
                nii_path = os.path.join(subdir.path,item.name + '_Merge.nii')
                hdf5_path = os.path.join(sub_save_path,ID + '.hdf5')
                # _,image = dicom_series_reader_without_postfix(item.path,'IM')
                _,image = dicom_series_reader(item.path)
                _, mask = nii_reader(nii_path)
                print(f'sample: {ID}, val max: {np.max(image)}, val min: {np.min(image)}, shape: {image.shape}')
                save_as_hdf5(image.astype(np.int16),hdf5_path,'image')
                save_as_hdf5(mask.astype(np.int16),hdf5_path,'mask')



if __name__ == "__main__":

    # convert image to numpy array and save as hdf5
    input_path = '../dataset/raw_data/'
    save_path = '../dataset/npy_data/'
    convert_to_npy(input_path,save_path)

  