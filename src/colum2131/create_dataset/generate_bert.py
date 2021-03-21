import pandas as pd
import numpy as np
from pathlib import Path
import torch
import transformers

from transformers import BertTokenizer
from tqdm import tqdm
tqdm.pandas()

class BertSequenceVectorizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'bert-base-multilingual-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128
        print(self.device)


    def vectorize(self, sentence : str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()
        
train = pd.read_csv('../../../data/train.csv')
test = pd.read_csv('../../../data/test.csv')
features_dir = Path('../../../features/colum2131/')

BSV = BertSequenceVectorizer()
for col in ['title', 'long_title','description', 'more_title', 'acquisition_credit_line']: 
    train_bert = train[col].fillna('nan').progress_apply(lambda x: BSV.vectorize(x))
    test_bert = test[col].fillna('nan').progress_apply(lambda x: BSV.vectorize(x))
    pd.DataFrame(train_bert.tolist(),columns=[f'{col}_bert_{i}' for i in range(768)]).to_pickle(features_dir / f'Bert_feature/bert_train_{col}.pickle')
    pd.DataFrame(test_bert.tolist(),columns=[f'{col}_bert_{i}' for i in range(768)]).to_pickle(features_dir / f'Bert_feature/bert_test_{col}.pickle')