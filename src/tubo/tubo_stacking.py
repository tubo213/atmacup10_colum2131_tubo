#!/usr/bin/env python
# coding: utf-8

# # Import

# In[1]:


import pandas as pd
import numpy as np
from module import stacking_models as models
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn import preprocessing
import glob


# # Define function

# In[2]:


def make_cv(n_split,seed):
    fold = StratifiedKFold(n_splits=n_split,shuffle=True,random_state=seed)
    bins = [1 if i > 0 else 0 for i in target.tolist()]
    return list(fold.split(train,bins))


# # Config

# In[3]:


INPUT_PATH = '../../../data/'
OUTPUT_PATH = '../../../'
CONFIG_PATH = '../config/'
ENSEMBLE_PATH = '../../../ensemble/'
SEED = 12
N_SPLIT = 5


# # Load Data

# In[4]:


target = np.log1p(pd.read_csv(INPUT_PATH+'train.csv')['likes'])
train = pd.read_csv(INPUT_PATH+'train.csv')
sub = pd.read_csv(INPUT_PATH+'atmacup10__sample_submission.csv')


# # 1層目

# In[5]:


#予測値呼び出し

tubo_train = pd.read_csv(ENSEMBLE_PATH+'tubo_train.csv')
colum_train = pd.concat([pd.read_csv(path) for path in sorted(glob.glob(ENSEMBLE_PATH+'colum2131_data/*train_model*'))], axis=1)
tubo_test = pd.read_csv(ENSEMBLE_PATH+'tubo_test.csv')
colum_test = pd.concat([pd.read_csv(path) for path in sorted(glob.glob(ENSEMBLE_PATH+'colum2131_data/*test*'))], axis=1)

stack_train_df = pd.concat([tubo_train,colum_train],axis=1)
stack_test_df = pd.concat([tubo_test,colum_test],axis=1)

#正規化
mm = preprocessing.StandardScaler()
mm.fit(pd.concat([stack_train_df,stack_test_df],axis=0))

stack_train_df = pd.DataFrame(mm.transform(stack_train_df),columns=stack_train_df.columns)
stack_test_df = pd.DataFrame(mm.transform(stack_test_df),columns=stack_test_df.columns)


# # 2層目
# Ridge,RandomForest,LightGBM,SVR,from sklearn.ensemble import ExtraTreesRegressorでstacking

# In[6]:


cv = make_cv(N_SPLIT,SEED)

model1 = models.Rid(0)
oof_preds_rid, test_preds_rid, evals_result_rid =model1.cv(
    target,stack_train_df,stack_test_df,cv
)


# In[7]:


model3 = models.Rdf(5,90)
oof_preds_rdf, test_preds_rdf, evals_result_rdf =model3.cv(
    target,stack_train_df,stack_test_df,cv
)


# In[8]:


lgbm_params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "num_leaves":2,
    "learning_rate":0.01,
    "verbose":-1}
model4 = models.Lgbm(lgbm_params)
oof_preds_lgb, test_preds_lgb, evals_result_lgb =model4.cv(
    target,stack_train_df,stack_test_df,cv
)


# In[9]:


model5 = models.SVR('linear',4)
oof_preds_svr, test_preds_svr, evals_result_svr =model5.cv(
    target,stack_train_df,stack_test_df,cv
)


# In[9]:


model6 = models.Extra(7,80)
oof_preds_ext, test_preds_ext, evals_result_ext =model6.cv(
    target,stack_train_df,stack_test_df,cv
)


# In[11]:


stack_train = pd.DataFrame([oof_preds_rid,oof_preds_lgb,oof_preds_svr,oof_preds_rdf,oof_preds_ext]).T 
stack_test = pd.DataFrame([test_preds_rid,test_preds_lgb,test_preds_svr,test_preds_rdf,test_preds_ext]).T 


# # 3層目
# 2層目の予測値をRidge,LightGBMでstacking

# In[12]:


stack_model1 = models.Rid(0)
oof_preds_stack_rid, test_preds_stack_rid, evals_result_stack_rid =stack_model1.cv(
    target,stack_train,stack_test,cv
)


# In[13]:


stack_model2 = models.Lgbm(lgbm_params)
oof_preds_stack_lgb, test_preds_stack_lgb, evals_result_stack_lgb =stack_model2.cv(
    target,stack_train,stack_test,cv
)


# In[14]:


stack_train_final = pd.DataFrame([oof_preds_stack_rid,oof_preds_stack_lgb]).T 
stack_test_final = pd.DataFrame([test_preds_stack_rid,test_preds_stack_lgb]).T 


# # 4層目
# 3層目の予測値をRidgeでstacking

# In[15]:


stack_model = models.Rid(0)
oof_preds_stack, test_preds_stack, evals_result_stack =stack_model.cv(
    target,stack_train_final,stack_test_final,cv
)


# # Submission

# In[16]:


sub['likes'] = np.expm1(np.where(test_preds_stack <=0,0,test_preds_stack))
sub.to_csv(OUTPUT_PATH+'submission.csv',index=False)

