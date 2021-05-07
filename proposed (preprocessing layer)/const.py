import tensorflow as tf
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt


class loss_plt(tf.keras.callbacks.Callback):
  def on_train_begin(self,logs={}):
    self.losses = []
    self.val_losses =[]
    self.accuracy = []
    self.val_accuracy =[]
    with open('logs.txt','w') as f:
      f.write('---------------------------Train Logs-------------------------------------')
      f.close()

  def on_epoch_end(self,epoch,logs={}):
    clear_output(wait=True)
    self.val_losses.append(logs.get('val_loss'))
    self.losses.append(logs.get('loss'))

    self.val_accuracy.append(logs.get('val_accuracy'))
    self.accuracy.append(logs.get('accuracy'))

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(self.val_losses,color="green",label="val_loss")
    plt.plot(self.losses,color="red",label="loss")
    plt.legend()
    plt.title("loss curve");

    plt.subplot(1,2,2)
    plt.plot(self.val_accuracy,color="green",label="val_accuracy")
    plt.plot(self.accuracy,color="red",label="accuracy")
    plt.legend()
    plt.title("accuracy curve");
    plt.tight_layout()
    plt.show()
    log_lines = f'epoch {epoch+1}/50 loss: {self.losses[-1]:.03f} accuracy: {self.accuracy[-1]:.04f} val_loss: {self.val_losses[-1]:.03f} val_accuracy:{self.val_accuracy[-1]:.04f}'
    print(log_lines)
    with open('logs.txt','a') as f:
      f.write('\n'+log_lines)
      f.close()
