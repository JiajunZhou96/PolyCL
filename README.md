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

## Run the model<br />
### 1. Pretraining for PolyCL with key parameters summarized in ```config.json```.
```
train.py
```
### 2. Transfer Learning with sample configurations described in ```config_tf_notebook.json```
```
transfer_learning.py
```
## Benchmarking other models<br />
Models available for benchmarking are stored in the ```./other_evals/``` directory.

### Pretrained Models
- **polyBERT**<br />
  Run ```tf_polybert.py```<br />
- **Transpolymer**<br />
  Put the folder of Transpolymer model "pretrain.pt" in the directory to be refered as ```"./model/Trasnpolymer/pretrain.pt"``` and run "```tf_transpolymer.py```<br /> 

### Supervised Models
- ```gnn.py```, ```morgan_nn.py```, ```rf.py```, ```xgb.py```

