# PolyCL

## Requirements<br />

```
# create a new environment
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
$ pip install tqdm
$ conda install -c conda-forge rdkit
```

## Run the model <br />
### 1. Pretraining 
Run with key parameters for the pretraining summarized in ```config.json```.
```
train.py
```
### 2. Transfer Learning 
Run with sample configurations described in ```config_tf_notebook.json```
```
transfer_learning.py
```
## Benchmarking other models <br />
Models available for benchmarking are stored in the ```./other_evals/``` directory.

### Pretrained Models
- #### polyBERT <br />
  - Run ```tf_polybert.py ```
- #### Transpolymer <br />
  - Download the model folder of Transpolymer "pretrain.pt" from https://github.com/ChangwenXu98/TransPolymer/tree/master/ckpt <br />
  - Put the folder to the directory ```"./model/Trasnpolymer/"``` to be referred to as ```"./model/Trasnpolymer/pretrain.pt"```
  - Run "```tf_transpolymer.py```<br /> 

### Supervised Models
- #### GNNs <br />
  - Adjust 
- ```gnn.py```, ```morgan_nn.py```, ```rf.py```, ```xgb.py```

