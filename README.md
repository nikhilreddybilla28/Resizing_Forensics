# Resizing_Forensics
- The experiments have been carried out using Tensorflow.keras with tensoflow version 2.4.1
- The training and testing have been done in google colaboratory
- Training of the model has been done on GPU
 

### File Descriptions
- The folder `proposed (preprocessing layer)` contains all files and findings for our proposed method
- `Proposed_Method.ipynb`: Contains the creation and training of the deep learning model.
- `Results.ipynb`: Contains the results i.e accuracy vs resizing factor and accuracy vs Quality Factor 2.
- `DataLoader.py` : Contains the generator required for loading data during model training
- `utilities.py`: Contains the data preparation algorithm
- `accuracyMatrix_1.py`: Contains the script for generating the accuracy vs quality factor pairs matrix
- `const.py`: Contains script to generate training curves.
- `model_1_3.hdf5`: Saved weights for the trained model.
- `logs_1_3.txt`: Contains the training logs for the model.
- `requirements.txt`: version of certain libraries required. Do !pip install requirements.txt
