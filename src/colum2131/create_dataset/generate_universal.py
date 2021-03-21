import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow_hub as hub
import tensorflow_text
import os
import tqdm
from tensorflow import keras
import texthero as hero
import ssl

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
ssl._create_default_https_context = ssl._create_unverified_context

features_dir = Path('../../../features/colum2131/')
train = pd.read_csv('../../../data/train.csv')
test = pd.read_csv('../../../data/test.csv')

text_cols = ['title','description','long_title','more_title','acquisition_credit_line']    
url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
embed = hub.load(url)

num = 15
tr_idxs = np.linspace(0,len(train),num,dtype=int)
te_idxs = np.linspace(0,len(test),num,dtype=int)

train.fillna('nan',inplace=True)
test.fillna('nan',inplace=True)
for col in text_cols:
    out_train = np.zeros([len(train),512])
    out_test = np.zeros([len(test),512])
    for i in tqdm.tqdm(range(1,num)):
        out_train[tr_idxs[i-1]:tr_idxs[i]] = embed(train[col].iloc[tr_idxs[i-1]:tr_idxs[i]]).numpy()
        out_test[te_idxs[i-1]:te_idxs[i]] = embed(test[col].iloc[te_idxs[i-1]:te_idxs[i]]).numpy()
        keras.backend.clear_session()
        
    out_train = pd.DataFrame(out_train).add_prefix(f'{col}_')
    out_test = pd.DataFrame(out_test).add_prefix(f'{col}_')

    out_train.to_pickle(features_dir / f'Universal_feature/train/{col}.pickle')
    out_test.to_pickle(features_dir / f'Universal_feature/test/{col}.pickle')