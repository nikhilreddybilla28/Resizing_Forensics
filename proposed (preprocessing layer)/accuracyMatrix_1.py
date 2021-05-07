from PIL import Image
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
import pandas as pd
import cv2

"""if __name__=="__main__":
  parser = ArgumentParser()
  parser.add_argument("--to-file",default="matrix1.csv")
  parser.add_argument("--test-dir",default="/content/Test/")
  parser.add_argument("--QF1",default = [50,60,70,80,90])
  parser.add_argument("--QF2",default = [50,60,70,80,90,99])
  parser.add_argument("--cand-RF",default=[0.6 ,0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3, 1.4])
  parser.add_argument("--test-RF",default=0.6)
  parser.add_argument("--model",default=None)
  args = parser.parse_args()
"""

def matrix(model,QF1 = [50,60,70,80,90],QF2 = [50,60,70,80,90,99], cand_RF =[0.6 ,0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3, 1.4],test_dir = '/content/Test/',to_file = "matrix.csv"):
  
  def softmax(arr):
    return np.exp(arr)/np.sum(np.exp(arr))
  def crop_image(img):
    mid = (int(img.shape[0]/2),int(img.shape[1]/2))
    return img[mid[0]-128:mid[0]+128,mid[1]-128:mid[1]+128]
  
  label = lambda img: cand_RF.index(float(img.split('.jpg')[0].split('rf_')[-1]))

  df = pd.DataFrame(columns=["50","60","70","80","90","99"],index=[50,60,70,80,90])
  preds=None
  
  for id,qf1 in enumerate(QF1):
    for qf2 in QF2:
      name = f'{qf1}_Q2_{qf2}'
      images = []
      Y_true = []

      for img in os.listdir(test_dir):
        clip = img.split('_rf_')[0].split('Q1_')[-1]
        if clip==name:
          #image = np.asarray(Image.open(os.path.join(path,img)).convert('L'))[:512,:512]
          image = cv2.cvtColor(cv2.imread(os.path.join(test_dir,img)),cv2.COLOR_BGR2GRAY)
          image = np.expand_dims(crop_image(image),axis=2).astype(np.float32)
          images.append(image)
          Y_true.append(label(img))
      
      preds = np.stack([np.argmax(softmax(p)) for p in model.predict(np.stack(images))])

      pair_acc = accuracy_score(Y_true,preds)
      df.loc[qf1,str(qf2)] = pair_acc
  print(preds)
  df.to_csv(to_file,index=False)
  return df

      

      
      



