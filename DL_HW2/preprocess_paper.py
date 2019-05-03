import json
import pandas as pd


'''
# only use with jupyter notebook in vscode
import os
if 'DL_HW2' not in os.getcwd():
    os.chdir(os.getcwd()+'/DL_HW2')
'''
df_accept = pd.read_excel("data/ICLR_accepted.xlsx", index_col=0)
np_accept = df_accept[0].values
df_reject = pd.read_excel("data/ICLR_rejected.xlsx", index_col=0)
np_reject = df_reject[0].values

train_data = []
test_data = []
for i, a in enumerate(np_accept):
    if i < 50:
        test_data.append({
            "title": a.split(),
            "label": 1
        })
    else:
        train_data.append({
            "title": a.split(),
            "label": 1
        })
for i, r in enumerate(np_reject):
    if i < 50:
        test_data.append({
            "title": r.split(),
            "label": 0
        })
    else:
        train_data.append({
            "title": r.split(),
            "label": 0
        })
with open('data/train.json', 'w') as f:
    for data in train_data:
        json.dump(data, f)
        f.write('\n')
with open('data/test.json', 'w') as f:
    for data in test_data:
        json.dump(data, f)
        f.write('\n')
