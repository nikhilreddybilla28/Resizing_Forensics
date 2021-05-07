import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

def crop_image(img):
    mid = (int(img.shape[0]/2),int(img.shape[1]/2))
    return img[mid[0]-128:mid[0]+128,mid[1]-128:mid[1]+128]

def getTestSet(self,path,rfs):
    images= []
    labels = []
    class_dict = dict(zip(rfs,range(len(rfs))))
    label = lambda x: class_dict[float(x.split('.jpg')[0].split('rf_')[-1])]
    print("Preparing set")
    for id,img in enumerate(os.listdir(path)):
      print('\r{:.02f}% '.format((id+1)*100/len(os.listdir(path))),end='',flush=True)
      
      image = cv2.cvtColor(cv2.imread(os.path.join(path,img)),cv2.COLOR_BGR2GRAY)
      image = np.expand_dims(crop_image(image),axis=2).astype(np.float32)

      images.append(image)
      labels.append(label(img))
    try:
      images = np.stack(images,axis=0)
      labels = np.stack(labels,axis=0)
    except:
      pass
    return images,labels

class DataLoader:
  def __init__(self,path=None,val_path= None,merge=True,rfs=None):
    """ 
    custom ImageDataGenerator Object
    ++++++++++++++++++++++++++++++++++++++++++
    path: path to directory of images

    Example:

    data = DataLoader(path="/content/gdrive/My Drive/Sync/UCR/Train)
    loader = data.flow(batch_size=64)
    model.fit_generator(loader)
    """
    self.path = path
    self.files = [f for f in glob.glob(os.path.join(self.path,'*.jpg'))]
    if val_path is not None:
      if merge==True:
        self.files+= [f for f in glob.glob(os.path.join(val_path,'*.jpg'))]
        np.random.shuffle(self.files)
        split_id = int(len(self.files)*0.3)
        self.val_files = self.files[:split_id]
        self.files = self.files[split_id:]
      else:
        self.val_files = [f for f in glob.glob(os.path.join(val_path,'*.jpg'))]
    else:
      np.random.shuffle(self.files)

    self.rfs = rfs
    self.class_dict = dict(zip(self.rfs,range(len(self.rfs))))
    self.label = lambda x: self.class_dict[float(x.split('.jpg')[0].split('rf_')[-1])]

  def crop_image(self,img):
    mid = (int(img.shape[0]/2),int(img.shape[1]/2))
    return img[mid[0]-128:mid[0]+128,mid[1]-128:mid[1]+128]

  def image_reader(self,files):
    for img in files:
      image = cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2GRAY)
      image = np.expand_dims(self.crop_image(image),axis=2).astype(np.float32)
      yield image,self.label(os.path.basename(img))

  def batch_generator(self,items,batch_size):
    a=[]
    i=0
    for item in items:
      a.append(item)
      i+=1

      if i%batch_size==0:
        yield a
        a=[]
    if len(a) is not 0:
      yield a
  
  def flow(self,batch_size):
    """
    flow from given directory in batches
    ==========================================
    batch_size: size of the batch
    """
    while True:
      for bat in self.batch_generator(self.image_reader(self.files),batch_size):
        batch_images = []
        batch_labels = []
        for im,im_label in bat:
          batch_images.append(im)
          batch_labels.append(im_label)
        batch_images = np.stack(batch_images,axis=0)
        batch_labels =  np.stack(batch_labels,axis=0)
        yield batch_images,batch_labels

  def getValSet(self):
    images= []
    labels = []
    print("Preparing set")
    for id,img in enumerate(self.val_files):
      print('\r{:.02f}% '.format((id+1)*100/len(self.val_files)),end='',flush=True)
      
      image = cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB)
      image = np.mean(image,axis=2)

      images.append(image)
      labels.append(self.label(os.path.basename(img)))
    try:
      images = np.stack(images,axis=0)
      labels = np.stack(labels,axis=0)
    except:
      pass
    return images,labels
  
  def val_flow(self,batch_size):
    """
    flow from given directory in batches
    ==========================================
    batch_size: size of the batch
    """
    while True:
      for bat in self.batch_generator(self.image_reader(self.val_files),batch_size):
        batch_images = []
        batch_labels = []
        for im,im_label in bat:
          batch_images.append(im)
          batch_labels.append(im_label)
        batch_images = np.stack(batch_images,axis=0)
        batch_labels =  np.stack(batch_labels,axis=0)
        yield batch_images,batch_labels
  
  


