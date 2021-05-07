import shutil
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import scipy.signal
import seaborn as sns
import tensorflow as tf
from tqdm.notebook import tqdm


def extract_images(img,crop_size=1024):
    images = []
    for i in range(crop_size,img.shape[0],crop_size):
        for j in range(crop_size,img.shape[1],crop_size):
            images.append(img[i-crop_size:i,j-crop_size:j,:])
    return images

def makeDirs():
  if os.path.isdir('/content/SCR'):
    shutil.rmtree('/content/SCR')
    
  os.mkdir('/content/SCR')
  os.mkdir('/content/SCR/Train')
  os.mkdir('/content/SCR/Test')

  if os.path.isdir('/content/dcr'):
    shutil.rmtree('/content/dcr')
    
  os.mkdir('/content/dcr')
  os.mkdir('/content/dcr/Train')
  os.mkdir('/content/dcr/Test')

def tamper(images=None,QF1=None,QF2=None,RF=None):
  print('\nPerforming Double Compression with tampering:')
  makeDirs()

  for id,image in enumerate(tqdm(images,desc="Performing Double Compression")):
    for q1 in QF1:
      name= f'im{id}_Q1_{q1}'
      cv2.imwrite('/content/SCR/Test/'+f'{name}.jpg',image,np.array([int(cv2.IMWRITE_JPEG_QUALITY),q1]))
      temp = cv2.imread(f'/content/SCR/Test/{name}.jpg')
      for q2 in QF2:
        for rf in RF:
          r_temp = cv2.resize(temp,(int(temp.shape[0]*rf),int(temp.shape[1]*rf)),interpolation=cv2.INTER_CUBIC)
          cv2.imwrite('/content/dcr/Test/'+f'{name}_Q2_{q2}_rf_{rf}.jpg',r_temp,np.array([int(cv2.IMWRITE_JPEG_QUALITY),q2]))    
  print('')
  shutil.rmtree('/content/SCR/') 

def PrepData(path=None, QF1=None, QF2=None,RF=None):
  #read uncompressed images and crop out patches of 256X256
  images = []
  
  for id,image in enumerate(tqdm(os.listdir(path),desc="Extracting Patches")):
    
    try:
      img = cv2.imread(os.path.join(path,image))
      temp = extract_images(img,crop_size=1024)
      images= images+temp
    except:
      pass
  #tamper with double compression
  tamper(images,QF1,QF2,RF)



