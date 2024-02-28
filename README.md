# PolyCL


## Run the model
1. Pretraining for PolyCL with key parameters summarized in ```config.json```.
```
train.py
```
2. Transfer Learning following the sample configurations described in ```config_tf_notebook.json```
```
transfer_learning.py
```



## Requirements<br />

```
$ conda create --name polycl python=3.9
$ conda activate polycl
# install requirements
#$ pip install numpy==1.21.5
#$ pip install pandas==1.3.3
#$ pip install scikit-learn==0.24.2
$ pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install transformers==4.20.1
$ pip install -U torchmetrics
$ pip install tensorboard
$ conda install -c conda-forge rdkit
```
