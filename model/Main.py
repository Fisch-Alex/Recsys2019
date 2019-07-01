import pandas as pd 
import numpy as np
import os 
from scipy.sparse import csr_matrix, hstack

from sklearn import preprocessing

my_dir = os.environ['Trivago']
os.chdir(my_dir)

from Frequency_model import Frequency_model

if os.environ['Trivago_debug']=='True':
    train_data = pd.read_csv("data/DEBUG_train.csv")
    test_data  = pd.read_csv("data/DEBUG_test.csv")
    meta_data  = pd.read_csv("data/DEBUG_metadata_updated.csv")
else:
    train_data = pd.read_csv("data/train.csv")
    test_data  = pd.read_csv("data/test.csv")
    meta_data  = pd.read_csv("data/metadata_updated.csv")

try:
    num_threads=int(os.environ['num_threads'])
except:
    num_threads=4
# Let's agree on this being our data 
train_instances = [x[1].reset_index(drop=True) for x in train_data.groupby("user_id")]
test_instances  = [x[1].reset_index(drop=True) for x in test_data.groupby("user_id")]


# This has to be tuned further
params = {
'task': 'train',
'boosting_type': 'gbdt',
'objective': 'lambdarank',
'lambda_l2' : 0.0037996,
'lambda_l1' : 190.0417685,
'metric': {'ndcg'},
'max_position': 4, ##how many ranks the lgbm cares about
#'metric': {'l2', 'auc', 'binary'},
'num_leaves': 92,
'bagging_fraction': 0.82191889,
'bagging_freq':10,
'max_depth': 30,
'max_bin':63,
'feature_fraction':0.6,
'min_data_in_leaf':73,
'learning_rate': 0.01,
'verbose': 10,
'output_model' : 'model/logs/model.txt', #Lets us load the model after
'metric_freq':5,
'num_threads': num_threads
}

base_name='Final'
if os.environ['Trivago_debug']=='True':
    num_round = 15
    base_name=base_name+'_DEBUG'
else:
    num_round = 17000

My_Model = Frequency_model(meta_data,params,num_round, num_threads=num_threads)

Tune=True
data_premade=False

if Tune:
    My_Model.fit(train_instances,test_instances=test_instances, Tune = Tune, data_premade=data_premade, K=5)

    num_round_new = My_Model.optimal_rounds

    print("Retraining on full Dataset num_round=%s --------------------------"%num_round_new)

    My_Model.change_num_round(num_round_new)

    data_premade = True
    My_Model.record_valid_preds(base_name)
My_Model.fit(train_instances,Tune=False, test_instances=test_instances, data_premade=data_premade) 

My_Model.predict(test_instances,out=False)
print(My_Model.timer)
My_Model.predictions_to_csv("%s.csv"%base_name)
My_Model.record_test_preds(base_name)
My_Model.log_model(base_name, newname=False)
