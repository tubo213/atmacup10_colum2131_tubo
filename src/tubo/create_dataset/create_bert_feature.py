import pandas as pd
import numpy as np
import torch
import transformers
from category_encoders import OrdinalEncoder
from transformers import BertTokenizer
from tqdm import tqdm
import re
tqdm.pandas()
import os
#kaeru bert のmutilingul version、前処理なしで抽出
#かえるclass
class BertSequenceVectorizer:
    def __init__(self,model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128


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
            return seq_out[0][0].cpu().detach().numpy() # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0][0].detach().numpy()

INPUT_PATH = '../../../data/'
OUTPUT_PATH = '../../../features/tubo/'

def main():
    train = pd.read_csv(INPUT_PATH + 'train.csv')
    test = pd.read_csv(INPUT_PATH + 'test.csv')
    text_col=['title','description','long_title','more_title','acquisition_credit_line']

    for col in text_col:
        train[col].fillna('nan',inplace=True)
        test[col].fillna('nan',inplace=True)

        b = BertSequenceVectorizer("bert-base-multilingual-cased")   
        train[col] = train[col].progress_apply(lambda x:b.vectorize(x))
        test[col] = test[col].progress_apply(lambda x:b.vectorize(x))

        pd.DataFrame(np.stack(train[col])).add_prefix('Bert_multi_').to_pickle(OUTPUT_PATH + f'Bert_multi/train/{col}.pickle')
        pd.DataFrame(np.stack(test[col])).add_prefix('Bert_multi_').to_pickle(OUTPUT_PATH + f'Bert_multi/test/{col}.pickle')   

if __name__ == '__main__':
    main()