import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import abstractmethod
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost
from catboost import Pool
from sklearn.metrics import mean_squared_error,accuracy_score,f1_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH='../../../data/'
cat_col = ['title_lang_ft','principal_maker','principal_or_first_maker','copyright_holder','acquisition_method']

#hakubihsin3さんのパクリ
class Base_Model(object):
    @abstractmethod
    def fit(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, features):
        raise NotImplementedError

    def cv(self, y_train, train_features, test_features, fold_ids,is_rmsle=True):
        test_preds = np.zeros(len(test_features))
        oof_preds = np.zeros(len(train_features))
        if(is_rmsle==False):
            test_preds = pd.DataFrame()

        for i_fold, (trn_idx, val_idx) in enumerate(fold_ids):

            x_trn = train_features.iloc[trn_idx]
            y_trn = y_train[trn_idx]
            x_val = train_features.iloc[val_idx]
            y_val = y_train[val_idx]

            model = self.fit(x_trn, y_trn, x_val, y_val)

            oof_preds[val_idx] = self.predict(model, x_val)
            if(is_rmsle):
                oof_score = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
                print('fold{}:RMSLE {}'.format(i_fold,oof_score))
                test_preds += self.predict(model, test_features) / len(fold_ids)
            else:
                oof_score = f1_score(y_val,np.round(oof_preds[val_idx]))
                print('fold{}:Acc{}'.format(i_fold,oof_score))
                test_preds['fold_{}'.format(i_fold)] = self.predict(model, test_features)


        if(is_rmsle):
            oof_score = np.sqrt(mean_squared_error(y_train, oof_preds))
            print('------------------------------------------------------')
            print(f'oof score: {oof_score}')
            print('------------------------------------------------------')

        else:
            oof_score = f1_score(y_train,np.round(oof_preds))
            print('------------------------------------------------------')
            print(f'oof score: {oof_score}')
            print('------------------------------------------------------')
            test_preds = test_preds.T.mode().loc[0]

        evals_results = {"evals_result": {
            "oof_score": oof_score,
            "n_data": len(train_features),
            "n_features": len(train_features.columns),
        }}

        return oof_preds, test_preds, evals_results

class Lgbm(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        self.models = []
        self.feature_cols = None
        
    def fit(self,x_train,y_train,x_valid,y_valid):
        lgb_train = lgb.Dataset(x_train,y_train)
        lgb_valid = lgb.Dataset(x_valid,y_valid)
        
        model = lgb.train(self.model_params,
            train_set=lgb_train,
            valid_sets=[lgb_valid],
            valid_names=['valid'],
            categorical_feature=cat_col,
            early_stopping_rounds=20,
            num_boost_round=10000,
            verbose_eval=False)
        self.models.append(model)
        return model
    
    def predict(self,model,features):
        self.feature_cols = features.columns
        return model.predict(features)

    def visualize_importance(self):
        feature_importance_df = pd.DataFrame()

        for i,model in enumerate(self.models):
            _df = pd.DataFrame()
            _df['feature_importance'] = model.feature_importance(importance_type='gain')
            _df['column'] = self.feature_cols
            _df['fold'] = i+1
            feature_importance_df = pd.concat([feature_importance_df,_df],axis=0,ignore_index=True)
        
        order = feature_importance_df.groupby('column').sum()[['feature_importance']].sort_values('feature_importance',ascending=False).index[:50]

        fig, ax = plt.subplots(2,1,figsize=(max(6, len(order) * .4), 14))
        sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance', order=order, ax=ax[0], palette='viridis')
        ax[0].tick_params(axis='x', rotation=90)
        ax[0].grid()
        fig.tight_layout()
        return fig,ax



class Cat(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        self.models = []
        self.feature_cols = None
        self.importance = None
    def fit(self,x_train,y_train,x_valid,y_valid):
        text_col = x_train.select_dtypes(include=object).columns.tolist()
        train_pool = Pool(x_train,
                          label=y_train,
                          cat_features=cat_col,
                          text_features=text_col)
        valid_pool = Pool(x_valid,
                          label=y_valid,
                          cat_features=cat_col,
                          text_features=text_col)
        
        model = CatBoost(self.model_params)
        model.fit(train_pool,
                  early_stopping_rounds=30,
                 plot=False,
                 use_best_model=True,
                 eval_set=[valid_pool],
                 verbose=False)
        self.models.append(model)
        return model
    
    def predict(self,model,features):
      self.feature_cols = features.columns
      pred = model.predict(features)
      return pred

    def visualize_importance(self):
        feature_importance_df = pd.DataFrame()

        for i,model in enumerate(self.models):
            _df = pd.DataFrame()
            self.importance = model.get_feature_importance()
            _df['feature_importance'] = model.get_feature_importance()
            _df['column'] = self.feature_cols.tolist()
            _df['fold'] = i+1
            feature_importance_df = pd.concat([feature_importance_df,_df],axis=0,ignore_index=True)
        
        order = feature_importance_df.groupby('column').sum()[['feature_importance']].sort_values('feature_importance',ascending=False).index[:50]

        fig, ax = plt.subplots(2,1,figsize=(max(6, len(order) * .4), 14))
        sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance', order=order, ax=ax[0], palette='viridis')
        ax[0].tick_params(axis='x', rotation=90)
        ax[0].grid()
        fig.tight_layout()
        return fig,ax
    
class Xgb(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        
    def fit(self,x_train,y_train,x_valid,y_valid):
        xgb_train = xgb.DMatrix(x_train,label=y_train)
        xgb_valid = xgb.DMatrix(x_valid,label=y_valid)
        
        evals = [(xgb_train,'train'),(xgb_valid,'eval')]
        
        model = xgb.train(self.model_params,
                         xgb_train,
                         num_boost_round=10000,
                         early_stopping_rounds=20,
                         evals=evals,
                         verbose_eval=False)
        
        return model
    
    def predict(self,model,features):
        return model.predict(xgb.DMatrix(features))

class Rid(Base_Model):
    def __init__(self):
      self.model = None
    def fit(self,x_train,y_train,x_valid,y_valid):
        model =Ridge(
            alpha=1, #L2係数
            max_iter=1000,
            random_state=10,
                              )
        model.fit(x_train,y_train)
        return model
      
    def predict(self,model,features):
      return model.predict(features)

    

class Rdf(Base_Model):
  def __init__(self,depth,n_est):
        self.depth=depth
        self.n_est=n_est
  def fit(self,x_train,y_train,x_valid,y_valid):
    model = RandomForestRegressor(max_depth=self.depth,n_estimators=self.n_est,random_state=0,n_jobs=-1)
    model.fit(x_train,y_train)
    return model

  def predict(self,model,features):
    pred = model.predict(features)
    return pred

from keras import models
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import backend as K
from keras.models import Model
class NN(Base_Model):
    def __init__(self,epoch,hidden_size):
      self.models = []
      self.epoch = epoch
      self.historys = []
      self.hidden_size = hidden_size

    def fit(self,x_train,y_train,x_valid,y_valid):
        model = self.build_model(x_train,self.hidden_size)
                
        history = model.fit(x_train.values,
                    y_train.values,
                    epochs=self.epoch,
                    batch_size=256,
                    validation_data=(x_valid.values,y_valid.values),
                    verbose=False
                    )
        
        self.historys.append(history.history)
        self.models.append(model)
        return model

    def predict(self,model,features):
      return model.predict(features)[0]
  
    def build_model(self,input_df,hidden_size):
        input_shape = input_df.shape
        model  = Sequential()
        model.add(Dense(128,activation='relu',input_shape=(input_shape[1],)))
        model.add(Dropout(0.7))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.7))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.7))
        model.add(Dense(hidden_size,activation='relu',name='hidden_layer'))
        model.add(Dense(1))
        model.compile(optimizer='adam',loss=self.rmse,metrics=[self.rmse])
        return model
        
    def rmse(self,y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
    def visualize_learning(self):
        loss = np.mean(np.array([his['loss'] for his in self.historys] ),axis=0)
        val_loss = np.mean(np.array([his['val_loss'] for his in self.historys] ),axis=0)
        epochs = range(1,len(loss)+1)
        
        plt.plot(epochs,loss,'bo',label='Trainig loss mean')
        plt.plot(epochs,val_loss,'b',label='Validation loss mean')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def get_hidden_output(self,train_df,test_df,fold_ids):
        train_hidden = np.zeros([len(train_df),self.hidden_size])
        test_hidden = np.zeros([len(test_df),self.hidden_size])

        for i_fold, (trn_idx, val_idx) in enumerate(fold_ids):
            model = self.models[i_fold]
            hidden_model = Model(
                inputs=model.input,
                outputs=model.get_layer('hidden_layer').output
            )
            
            train_hidden[val_idx] = hidden_model.predict(train_df.iloc[val_idx,:])
            test_hidden += hidden_model.predict(test_df)/len(fold_ids)
            
        return pd.DataFrame(train_hidden).add_prefix('Hidden_'),pd.DataFrame(test_hidden).add_prefix('Hidden_')
    
class NN_stack(Base_Model):
    def __init__(self,epoch,hidden_size):
      self.models = []
      self.epoch = epoch
      self.historys = []
      self.hidden_size = hidden_size

    def fit(self,x_train,y_train,x_valid,y_valid):
        model = self.build_model(x_train,self.hidden_size)
                
        history = model.fit(x_train.values,
                    y_train.values,
                    epochs=self.epoch,
                    batch_size=256,
                    validation_data=(x_valid.values,y_valid.values),
                    verbose=False
                    )
        
        self.historys.append(history.history)
        self.models.append(model)
        return model

    def predict(self,model,features):
      return model.predict(features)[0]
  
    def build_model(self,input_df,hidden_size):
        input_shape = input_df.shape
        model  = Sequential()
        model.add(Dense(32,activation='relu',input_shape=(input_shape[1],)))
        model.add(Dropout(0.5))
        model.add(Dense(32,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam',loss=self.rmse,metrics=[self.rmse])
        return model
        
    def rmse(self,y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
    def visualize_learning(self):
        loss = np.mean(np.array([his['loss'] for his in self.historys] ),axis=0)
        val_loss = np.mean(np.array([his['val_loss'] for his in self.historys] ),axis=0)
        epochs = range(1,len(loss)+1)
        
        plt.plot(epochs,loss,'bo',label='Trainig loss mean')
        plt.plot(epochs,val_loss,'b',label='Validation loss mean')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()