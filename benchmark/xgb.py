import os
os.chdir(os.pardir) # to the parent dir

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import utils
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

seed = 72
utils.set_seed(seed)
dataset = pd.read_csv("./datasets/Eea.csv")

def smiles_to_fp(smiles, radius=2, nBits=512):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)

dataset['fingerprints'] = dataset['smiles'].apply(lambda x: smiles_to_fp(x))
X = np.array(list(dataset['fingerprints']))  # Features
y = dataset['value'].values # Labels

#param_grid = {
#    'max_depth': [3, 6, 9],
#    'min_child_weight': [1, 3],
#    'subsample': [0.8, 1.0],
#}

kf = KFold(n_splits= 5, shuffle=True, random_state=seed)
model = xgb.XGBRegressor(max_depth = 9, min_child_weight = 3, colsample_bytree=0.8, learning_rate=0.2, random_state=seed)

rmses = []
r2s = []

# K-Fold Cross-Validation
for train_index, test_index in kf.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rmses.append(rmse)
    r2s.append(r2)

# Calculate and print the average RMSE and R2
average_rmse = np.mean(rmses)
average_r2 = np.mean(r2s)
std_dev_rmse = np.std(rmses)
std_dev_r2 = np.std(r2s)

print(f'Average test RMSE: {average_rmse:.4f}')
print(f'RMSE Standard Deviation: {std_dev_rmse:.4f}')
print(f'Average test R2: {average_r2:.4f}')
print(f'R2 Standard Deviation: {std_dev_r2:.4f}')
