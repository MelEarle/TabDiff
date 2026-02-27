import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, roc_auc_score, root_mean_squared_error, accuracy_score

parser = argparse.ArgumentParser(description='Evaluate Imputation')
parser.add_argument('--dataname', type=str)
parser.add_argument('--exp_name', type=str, default='tabsyn')
args = parser.parse_args()

curr_dir = os.path.dirname(os.path.abspath(__file__))
dataname = args.dataname
exp_name = args.exp_name

with open(f'{curr_dir}/data/{dataname}/info.json', 'r') as f:
    info = json.load(f)

task_type = info['task_type']
real_path = f'synthetic/{dataname}/test.csv'
real_data = pd.read_csv(real_path)

num_target_col_idx = info.get('num_target_col_idx', [])
cat_target_col_idx = info.get('cat_target_col_idx', [])
if len(num_target_col_idx) == 0 and len(cat_target_col_idx) == 0:
    if task_type == 'regression':
        num_target_col_idx = info['target_col_idx']
    else:
        cat_target_col_idx = info['target_col_idx']

target_cols_num = [real_data.columns[i] for i in num_target_col_idx]
target_cols_cat = [real_data.columns[i] for i in cat_target_col_idx]

syn_tables = []
for i in range(50):
    syn_path = f'impute/{dataname}/{exp_name}/{i}.csv'
    syn_tables.append(pd.read_csv(syn_path))

if len(target_cols_cat) > 0:
    print("Categorical targets:")
    cat_acc, cat_f1 = [], []
    for target_col in target_cols_cat:
        encoder = OneHotEncoder()
        real_target = real_data[target_col].to_numpy().reshape(-1, 1)
        real_y = encoder.fit_transform(real_target).toarray()

        syn_y = []
        for syn_data in syn_tables:
            target = syn_data[target_col].to_numpy().reshape(-1, 1)
            syn_y.append(encoder.transform(target).toarray())

        syn_y_prob = np.stack(syn_y).mean(0)
        pred = syn_y_prob.argmax(axis=1)
        truth = real_y.argmax(axis=1)

        acc = accuracy_score(truth, pred)
        micro_f1 = f1_score(truth, pred, average='micro')
        cat_acc.append(acc)
        cat_f1.append(micro_f1)
        print(f"  {target_col}: ACC={acc:.4f}, F1={micro_f1:.4f}")

        if real_y.shape[1] > 1:
            try:
                auc = roc_auc_score(real_y, syn_y_prob, average='micro', multi_class='ovr')
                print(f"  {target_col}: AUC={auc:.4f}")
            except Exception:
                pass

    print(f"Categorical macro ACC: {np.mean(cat_acc):.4f}")
    print(f"Categorical macro F1: {np.mean(cat_f1):.4f}")

if len(target_cols_num) > 0:
    print("Numerical targets:")
    rmses = []
    for target_col in target_cols_num:
        y_test = real_data[target_col].to_numpy().astype(float)
        syn_vals = []
        for syn_data in syn_tables:
            syn_vals.append(syn_data[target_col].to_numpy().astype(float))
        pred = np.stack(syn_vals).mean(0)
        rmse = root_mean_squared_error(y_test, pred)
        rmses.append(rmse)
        print(f"  {target_col}: RMSE={rmse:.4f}")
    print(f"Numerical mean RMSE: {np.mean(rmses):.4f}")
