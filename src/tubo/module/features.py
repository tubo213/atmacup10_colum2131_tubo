import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import transformers
from category_encoders import OrdinalEncoder
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from pandarallel import pandarallel
from PIL import ImageColor
from geopy.geocoders import Nominatim
import re
from sklearn.decomposition import NMF, PCA, TruncatedSVD
import pickle5
from sklearn.model_selection import StratifiedKFold
from category_encoders.target_encoder import TargetEncoder
from datetime import datetime as dt
from umap import UMAP
import models
pandarallel.initialize()
tqdm.pandas()

INPUT_PATH = '../../../data/'
FEATURE_PATH = '../../../features/tubo/'
SEED = 12

train = pd.read_csv(INPUT_PATH+'train.csv')
test = pd.read_csv(INPUT_PATH+'test.csv')
whole_df = pd.concat([train,test],axis=0)
text_cols = ['title','description','long_title','more_title','acquisition_credit_line']
use_col = ['dating_sorting_date','dating_period','dating_year_early','dating_year_late','make_period']
cat_col = ['title_lang_ft','principal_maker','principal_or_first_maker','copyright_holder','acquisition_method']

#そのまま使う特徴量
def identity_func(df):
    return df[use_col]

#日時データを数値に変換
def datingdata(df):
    out_df = pd.DataFrame()
    _df = df['acquisition_date'].apply(lambda x:str(x).split('T')[0])

    out_df['year'] =  _df.apply(lambda x:int(str(dt.strptime(x, '%Y-%m-%d')).split('-')[0]) if x != 'nan' else x
)
    out_df['month'] =  _df.apply(lambda x:int(str(dt.strptime(x, '%Y-%m-%d')).split('-')[1]) if x != 'nan' else x
)
    out_df.replace({'nan':np.NaN},inplace=True)
    return out_df.add_prefix('acquisition_')

#kaeru bert
class BertSequenceVectorizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'bert-base-uncased'
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

#gotoさんのBaseBlock!!
class BaseBlock(object):
    def fit(self,input_df,y=None):
        return self.transform(input_df)
    
    def transform(self,input_df):
        raise NotImplementedError()


class WrapperBlock(BaseBlock):
    def __init__(self,function):
        self.function=function

    def transform(self,input_df):
        return self.function(input_df)

#ラベルエンコーディングするところ
class LabelEncodingBlock(BaseBlock):
    def __init__(self,cols:list,whole_df):
        self.cols = cols
        self.whole_df = whole_df
        self.oe = None

    def fit(self,input_df):
        self.whole_df[self.cols].fillna('NAN',inplace=True)
        oe = OrdinalEncoder(cols=self.cols,handle_unknown='inpute')
        oe.fit(self.whole_df[self.cols])
        self.oe = oe
        return self.transform(input_df)

    def transform(self, input_df):
        input_df[self.cols].fillna('NAN',inplace=True)
        return self.oe.transform(input_df[self.cols]) 


#カウントエンコーディング
class CountEncodingBlock(BaseBlock):
    def __init__(self,col):
        self.col = col

    def transform(self,input_df):
        df = input_df.groupby(self.col)['object_id'].transform('count')
        df.name = ('CE' + self.col + '_')
        return df

#ターゲットエンコーディング(10回以上出現するやつだけ。cvみたいにしてないから多少リークしてる。smoothing=0.1)
class TargetEncodingBlock(BaseBlock):
    def __init__(self,data_name):
        self.data_name = data_name
        self.meta = None

    def fit(self,input_df):
        df = pd.read_csv(INPUT_PATH+f'{self.data_name}.csv')
        train  = pd.read_csv(INPUT_PATH+'train.csv')
        vc = df['name'].value_counts()
        idx = vc[vc>=10].index
        df = df[df['name'].isin(idx)]
    
        df = df.merge(train[['object_id','likes']],on='object_id',how='left')

        self.meta = pd.concat(
            [df['object_id'],
            TargetEncoder(smoothing=0.1).fit_transform(df['name'],
            np.log1p(df['likes']))],
            axis=1).groupby('object_id').agg(['mean','sum','max','min','std'])
        self.meta.columns = self.meta.columns.droplevel(0)

        return self.transform(input_df)

    def transform(self,input_df):
        return pd.merge(input_df['object_id'],self.meta,on='object_id',how='left')\
            .drop(columns='object_id').add_prefix(f'TE_{self.data_name}_')

#kyouheiさんのcountry特徴量
class CountryBlock(BaseBlock):
    def __init__(self):
        self.meta = None
    
    def fit(self,input_df):
#kyohei_featureの作り方
#        production_place = pd.read_csv(INPUT_PATH + 'production_place.csv')
#        place_list = production_place['name'].unique()
#        country_dict = {}
#        for place in tqdm(place_list):
#            try:
#                country = self.place2country(place)
#                country_dict[place] = country
#            except:
#                # 国名を取得できない場合はnan
#                print(place)
#        country_dict[place] = np.nan
#        production_place['country_name'] = production_place['name'].map(country_dict)   
#        self.meta = pd.crosstab(production_place['object_id'],production_place['country_name']).reset_index()
        self.meta = pd.read_csv(FEATURE_PATH+'country.csv')
        return self.transform(input_df)

    def transform(self,input_df):
        return pd.merge(input_df['object_id'],self.meta,on='object_id',how='left').drop(columns='object_id')

    def place2country(self,address):
        geolocator = Nominatim(user_agent='sample', timeout=200)
        loc = geolocator.geocode(address, language='en')
        coordinates = (loc.latitude, loc.longitude)
        location = geolocator.reverse(coordinates, language='en')
        country = location.raw['address']['country']
        return country

#object_collectionをクロスタブにしてPCA
class CollectBlock(BaseBlock):
    def __init__(self,num):
        self.meta = None
        self.num = num
    
    def fit(self,input_df):
        collect = pd.read_csv(INPUT_PATH+'object_collection.csv')
        tab = pd.crosstab(collect['object_id'],collect['name'])
        pca = PCA(self.num,random_state=SEED)
        self.meta = pd.DataFrame(pca.fit_transform(tab),index=tab.index)
        return self.transform(input_df)

    def transform(self, input_df):
        return pd.merge(input_df['object_id'],self.meta,on='object_id',how='left').drop(columns='object_id')\
            .add_prefix('Collect_')

#historical_person出現回数上位10人に絞ってクロスタブ
class PersonBlock(BaseBlock):
    def __init__(self):
        self.meta = None
    
    def fit(self,input_df):
        person = pd.read_csv(INPUT_PATH+'historical_person.csv')
        person['name'] = person['name'].apply(lambda x : re.sub('[\[\]\,\)\(\.\}\{\-]','',x))
        names = person['name'].value_counts().head(10).index
        self.meta = pd.crosstab(person['object_id'],person['name'])[names].reset_index()
        return self.transform(input_df)

    def transform(self, input_df):
        return pd.merge(input_df['object_id'],self.meta,on='object_id',how='left').drop(columns='object_id')
        
#techniqueをクロスタブにしてPCA
class TechBlock(BaseBlock):
    def __init__(self,dim):
        self.dim = dim
        self.meta = None

    def fit(self,input_df):
        tech = pd.read_csv(INPUT_PATH+'technique.csv')
        pca = PCA(self.dim,random_state=SEED)
        tab = pd.crosstab(tech['object_id'],tech['name'])
        self.meta = pd.DataFrame(pca.fit_transform(tab),index=tab.index)
        return self.transform(input_df)

    def transform(self, input_df):
        return pd.merge(input_df['object_id'],self.meta,on='object_id',how='left').drop(columns='object_id')

#sub_titleをmmに変換、面積や体積も追加
class SubtoSizeBlock(BaseBlock):
    def __init__(self):
        return 

    def transform(self,input_df):
        for axis in ['h', 'w', 't', 'd']:
            column_name = f'size_{axis}'
            size_info = input_df['sub_title'].str.extract(r'{} (\d*|\d*\.\d*)(cm|mm)'.format(axis)) # 正規表現を使ってサイズを抽出
            size_info = size_info.rename(columns={0: column_name, 1: 'unit'})
            size_info[column_name] = size_info[column_name].replace('', np.nan).astype(float) # dtypeがobjectになってるのでfloatに直す
            size_info[column_name] = size_info.apply(lambda row: row[column_name] * 10 if row['unit'] == 'cm' else row[column_name], axis=1) # 　単位をmmに統一する
            input_df[column_name] = size_info[column_name] # trainにくっつける
        input_df['area'] = input_df['size_h']*input_df['size_w']
        input_df['volume'] = input_df['area']*input_df['size_d']
        input_df['H-W'] = input_df['size_h'] - input_df['size_w']
        return input_df[[ 'size_h', 'size_w', 'size_t', 'size_d','area','volume','H-W']]

#前処理しないでtfid
class TfidBlock(BaseBlock):
    def __init__(self,whole_df:pd.DataFrame,col,feature_size:int):
        self.whole_df = whole_df
        self.size = feature_size
        self.col = col
    def transform(self,input_df):
        tfid = TfidfVectorizer(max_features=self.size)
        tfid.fit(self.whole_df[self.col])
        output_df  = tfid.transform(input_df[self.col])
        output_df= pd.DataFrame(output_df.toarray(),columns=tfid.get_feature_names())
        return output_df.add_prefix('Tfid_{}_'.format(self.col))

#テキストの長さ
class StrCountBlock(BaseBlock):
    def __init__(self,col:str):
        self.col = col
    
    def transform(self,input_df):
        out_df = pd.DataFrame()
        out_df[self.col] = input_df[self.col].parallel_apply(lambda x:len(x))
        return out_df.add_prefix('SCE_')

#colorのpercent上位n個を持ってくる
class ColorBlock(BaseBlock):
    def __init__(self,num):
        self.meta = None
        self.num = num
    
    def fit(self,input_df):
        color_df = pd.read_csv(INPUT_PATH+'color.csv')
        color_df[['R','G','B']] = pd.DataFrame(color_df['hex'].str.strip().map(ImageColor.getrgb).values.tolist(), columns=['R', 'G', 'B'])
        self.meta = color_df.groupby('object_id')['percentage','R','G','B'].apply(lambda x:x.sort_values(['percentage'],ascending=False).iloc[self.num]).reset_index()
        return self.transform(input_df)
    
    def transform(self, input_df):
        return pd.merge(input_df['object_id'],self.meta,on='object_id',how='left')\
            .drop(columns='object_id').add_suffix('_' + str(self.num))

#Paletteをobject_idで集約して統計量を計算
class PaletteBlock(BaseBlock):

    def __init__(self,whole_df):
        self.meta = None
        self.whole_df = whole_df['object_id']
    def fit(self,input_df):
        palette = pd.read_csv(INPUT_PATH+'palette.csv')
        max_palette = palette.groupby('object_id')['ratio'].max().reset_index()
        max_palette = pd.merge(max_palette, palette, on=['object_id','ratio'], how='left').rename(
            columns={"ratio":"max_ratio", "color_r":"max_palette_r", "color_g":"max_palette_g","color_b":"max_palette_b"})  
        max_palette = max_palette.loc[max_palette["object_id"].drop_duplicates().index.tolist()].reset_index()  # 同じidでmax ratioが同じものは削除

        mean_palette = palette.copy()
        mean_palette["color_r"] = palette["ratio"] * palette["color_r"]
        mean_palette["color_g"] = palette["ratio"] * palette["color_g"]
        mean_palette["color_b"] = palette["ratio"] * palette["color_b"]
        mean_palette = mean_palette.groupby("object_id").sum().reset_index().rename(
            columns={"color_r":"mean_palette_r", "color_g":"mean_palette_g","color_b":"mean_palette_b"})

        var_palette = palette.groupby('object_id').std()

        self.meta = pd.merge(self.whole_df,max_palette,on='object_id',how='left')
        self.meta = pd.merge(self.meta,mean_palette,on='object_id',how='left')
        self.meta = pd.merge(self.meta,var_palette,on='object_id',how='left')
        return self.transform(input_df)

    def transform(self,input_df):
        return pd.merge(input_df['object_id'],self.meta,how='left',on='object_id').drop(columns=['object_id','index'])

#materialをクロスタブしてPCA
class MaterialBlock(BaseBlock):
    def __init__(self,num):
        self.meta = None
        self.num = num

    def fit(self,input_df):
        pca =PCA(self.num,random_state=SEED)
        material = pd.read_csv(INPUT_PATH+'material.csv')
        tab = pd.crosstab(material['object_id'],material['name'])
        self.meta = pd.DataFrame(pca.fit_transform(tab),index=tab.index)
        return self.transform(input_df)

    def transform(self,input_df):
        return pd.merge(input_df['object_id'],self.meta,on='object_id',how='left').drop(columns='object_id')\
            .add_prefix('Material')

#principal_makerをクロスタブしてPCA
class PrincipalMakerBlock(BaseBlock):
    def __init__(self,dim,col):
        self.meta = None
        self.dim = dim
        self.col = col

    def fit(self,input_df):
        df = pd.read_csv(INPUT_PATH+'principal_maker.csv')
        pca = PCA(self.dim,random_state=SEED)
        tab = pd.crosstab(df['object_id'],df[self.col])
        self.meta = pd.DataFrame(pca.fit_transform(tab),index=tab.index)
        return self.transform(input_df)

    def transform(self,input_df):
        return pd.merge(input_df['object_id'],self.meta,on='object_id',how='left').drop(columns='object_id')\
            .add_prefix(f'Principal{self.col}_')

    

#kaeru bertで作ったやつをPCA
class BertMultiBlock(BaseBlock):
    def __init__(self,col,dim):
        self.col = col
        self.dim = dim
        self.pca = None

    def fit(self,input):
        train_df = pd.read_pickle(FEATURE_PATH + f'Bert_multi/train/{self.col}.pickle')
        test_df = pd.read_pickle(FEATURE_PATH + f'Bert_multi/test/{self.col}.pickle')
        whole = pd.concat([train_df,test_df],axis=0)
        pca = PCA(self.dim,random_state=SEED)
        pca.fit(whole)
        self.pca = pca
        
        train_df = pd.DataFrame(pca.transform(train_df))
        return train_df.add_prefix(f'multi_{self.col}_')

    def transform(self,input):
        test_df = pd.read_pickle(FEATURE_PATH + f'/Bert_multi/test/{self.col}.pickle')
        test_df = pd.DataFrame(self.pca.transform(test_df))
        return test_df.add_prefix(f'multi_{self.col}_')

#Universal Sentence Encoderで抽出したやつPCA
class UniversalBlock(BaseBlock):
    def __init__(self,col,dim):
        self.col = col
        self.dim = dim
        self.pca = None

    def fit(self,input):
        train_df = pd.read_pickle(FEATURE_PATH + f'Universal_feature/train/{self.col}.pickle')
        test_df = pd.read_pickle(FEATURE_PATH + f'Universal_feature/test/{self.col}.pickle')
        whole = pd.concat([train_df,test_df],axis=0)
        pca = PCA(self.dim,random_state=SEED)
        pca.fit(whole)
        self.pca = pca
        
        train_df = pd.DataFrame(pca.transform(train_df))
        return train_df.add_prefix(f'Universal_{self.col}_')

    def transform(self,input):
        test_df = pd.read_pickle(FEATURE_PATH + f'Universal_feature/test/{self.col}.pickle')
        test_df = pd.DataFrame(self.pca.transform(test_df))
        return test_df.add_prefix(f'Universal_{self.col}_')

#Araiさんの自己なんとかかんとか学習の特徴量、PCA
class AraiBlock(BaseBlock):
    def __init__(self,dim):
        self.dim = dim

    def fit(self,input_df):
        arai_feature = pd.read_pickle(FEATURE_PATH + 'arai_feature.pickle')
        pca = PCA(self.dim,random_state=SEED)
        self.meta = pd.DataFrame(
            pca.fit_transform(arai_feature),
            index = arai_feature.index
        )
        self.meta.index.name = 'object_id'
        return self.transform(input_df)

    def transform(self, input_df):
        return pd.merge(input_df['object_id'],self.meta,on='object_id',how='left')\
            .drop(columns='object_id').add_prefix('Arai_')

#yukiさんのpalette2imgをeffnetで特徴抽出してPCA
class Color2VecBlock(BaseBlock):
    def __init__(self,dim):
        self.dim = dim
        self.meta = None

    def fit(self,input_df):
        with open(FEATURE_PATH + 'calor2vec.pickle', "rb") as fh:
            df = pickle5.load(fh)
        pca = PCA(self.dim,random_state=SEED)
        self.meta = pd.DataFrame(pca.fit_transform(df),index=df.index).reset_index()
        self.meta.rename(columns={'index':'object_id'},inplace=True)
        return self.transform(input_df)

    def transform(self,input_df):
        out_df = pd.merge(input_df['object_id'],self.meta,on='object_id',how='left')\
            .drop(columns='object_id').add_prefix('Eff_')
        return out_df

#gotoさんのパクリ         
from contextlib import contextmanager
from time import time

@contextmanager
def timer(logger=None,format_str='{:.3f}[s]',prefix=None,suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time()
    yield
    d = time()-start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)
        
from tqdm import tqdm

def get_function(block,is_train):
    s = mapping ={
        True:'fit',
        False:'transform'
    }.get(is_train)
    return getattr(block,s)

def to_feature(input_df,blocks,is_train=False):
    out_df = pd.DataFrame()
    
    for block in tqdm(blocks,total=len(blocks)):
        func = get_function(block,is_train)
        
        with timer(prefix='create' + str(block) + ' '):
            _df = func(input_df)
        
        assert len(_df) == len(input_df),func._name_
        out_df = pd.concat([out_df,_df],axis=1)
    return out_df