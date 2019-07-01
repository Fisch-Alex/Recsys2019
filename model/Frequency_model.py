import pandas as pd 
from collections import Counter
import numpy as np
import lightgbm as lgb
from custom_eval_fns import custom_eval
from time import time
import os 
from Filterfunctions import gen_filter_dict
import sys

my_dir = os.environ['Trivago']
os.chdir(my_dir)

from Data_Transformations import my_train_splitter, my_test_splitter, cv_idx_gen, my_train_from_test_splitter, tt_split
from Train_Binarisers import Train_Binarisers
from Feature_extraction import Feature_extraction
from Weights import train_weights

class Frequency_model:

	def __init__(self,meta_info,params,num_round, num_threads, weights_flag=False):
		self.meta_info      = meta_info
		self.meta_info.fillna({'properties_reduced':''}, inplace=True)
		self.meta_info['properties_reduced']=self.meta_info['properties_reduced'].apply(lambda x: x.split('|'))
		self.params         = params
		self.num_threads    = num_threads
		self.num_round      = num_round 
		self.valid_MRR      = None
		self.optimal_rounds = None
		self.fname = None
		self.timer = {'make_dataset':[],
				'fit':[],
				'fit_validation':[],
				'predict':[]}
		self.weights_flag=weights_flag

	def change_num_round(self,num_round):
		self.num_round  = num_round 

	def change_parameters(self,params):
		self.params  = params 

	def make_dataset(self,train_instances, test_instances = None, K=5):
		"""Returns an LGB ready dataset and a evaluation function for MRR if asked for"""
		t = time()
		# Preprocessing 

		train_instances_expanded = [my_train_splitter(x) for x in train_instances]

		if test_instances is not None:
			test_train_instances_expanded = [my_train_from_test_splitter(x) for x in test_instances]
			train_instances_expanded = train_instances_expanded + test_train_instances_expanded
			

		train_history     = [z for x in train_instances_expanded for z in x["features"]]
		train_suggestions = [z for x in train_instances_expanded for z in x["suggestions"]]
		train_response    = [z for x in train_instances_expanded for z in x["responses"]]

		Filter = [ii for ii in range(len(train_response)) if train_response[ii] in train_suggestions[ii]]

		train_history     = [train_history[ii] for ii in Filter]
		train_suggestions = [train_suggestions[ii] for ii in Filter]
		train_response    = [train_response[ii] for ii in Filter]
		
		train_actual_response       = [a for b in [[ 1 if x == train_response[ii] else 0 for x in train_suggestions[ii] ] for ii in range(len(train_history))] for a in b]
		train_suggestion_identifier = [a for b in train_suggestions for a in b]
		train_userdaf_identifier    = [ii for ii in range(len(train_history)) for a in range(len(train_suggestions[ii]))]
		# Here is probably the moment to subsample!
                ##Will be ordered so no need to check
		count_in_instance = Counter(train_userdaf_identifier)
		query_count = [count_in_instance[i] for i in range(max(count_in_instance)+1)]
		self.query_count = query_count
		# Train Binarisers 
		print("Training Binarisers")

		self.Binarisers = Train_Binarisers(train_instances,self.meta_info)
		
		item_popularity_dict = {"total":0}
		item_suggestion_dict = {"total":0}

		if os.environ['Trivago_debug']=='True':
			train_file = "data/DEBUG_train.csv"
			test_file  = "data/DEBUG_test.csv"
			thresh=10
		else:
			train_file = "data/train.csv"
			test_file  = "data/test.csv"
			thresh=40
		print('generating filter dictionary')	
		self.filter_dict = gen_filter_dict(train_file,test_file, thresh)
		print("Extracting Features")


		X, self.item_popularity_dict, self.item_suggestion_dict = Feature_extraction(train_history,train_suggestions,train_suggestion_identifier,train_userdaf_identifier,self.Binarisers,self.meta_info,self.filter_dict,item_popularity_dict,item_suggestion_dict,train_actual_response)
		self.num_feat = X.shape[1]
			
		#user_ids = [train_history[ii]["user_id"][0] for ii in train_userdaf_identifier]
		tsub = time()
		#d_train, d_valid = tt_split(X, train_actual_response, query_count,K, stratify=user_ids)
		if test_instances is not None and self.weights_flag:
			self.weights=np.array(train_weights(train_history, test_instances, train_userdaf_identifier))
			d_train, d_valid, self.train_weights, self.test_weights , self.validqc= tt_split(X, train_actual_response, query_count,K,weights= self.weights)
		else:
			d_train, d_valid, self.validqc = tt_split(X, train_actual_response, query_count,K)
		d_full   = lgb.Dataset(X,  label=train_actual_response, group=query_count)
		
		self.timer["Validationsplitting and Full data"]= time()-tsub
		self.watchlist  = [d_train, d_valid, d_full]
		self.timer["make_dataset"].append(time()-t)
		
	def fit(self,train_instances, test_instances = None, Tune=False, K=5, data_premade=False):

		#Don't record time in make dataset as part of fit
		if not data_premade:
			self.make_dataset(train_instances, test_instances = test_instances, K=K)
		t=time()
		if Tune: 
			self.model = lgb.train(self.params, self.watchlist[0], self.num_round, valid_sets = self.watchlist[:2], early_stopping_rounds = 500) 
			self.timer['fit_validation'].append(time()-t)
			self.optimal_rounds = self.model.current_iteration()
			self.eval_MRR(self.watchlist[1])
		else:
			self.model = lgb.train(self.params, self.watchlist[2], self.num_round)
			self.timer['fit'].append(time()-t)
	def eval_MRR(self, valid_dataset, print_flag=True, rank_file=None, extra_append=''):
		"""
		Valid dataset must be either raw data pre lgb.Dataset, or a lgb.Dataset with free_raw_data=False
		"""
		preds     = self.model.predict(valid_dataset.data)
		eval_fn   = custom_eval()		
		if self.weights_flag:
			RR_valid, weights = eval_fn.MRR_post(preds, train_data=valid_dataset)
		else: 
			RR_valid  = eval_fn.MRR_post(preds, train_data=valid_dataset)
		MRR_valid = np.mean(RR_valid)
		self.valid_MRR = MRR_valid
		if print_flag:
			print("Valid_MRR: %s, %s"%(MRR_valid, extra_append))
			if self.weights_flag:
				try:
					print('weighted_MRR: %s' % (np.average(np.array(RR_valid), weights=weights, returned=True)[0]))
				except:
					print("weighted MRR failed")
		if rank_file is not None:
			valid_df=pd.DataFrame(RR_valid)
			valid_df.to_csv(rank_file)
		return(MRR_valid)

	def predict(self,test_instances,out=True):

		t=time()
		# Preprocessing 

		test_instances_expanded = [my_test_splitter(x) for x in test_instances]

		test_history     = [z for x in test_instances_expanded for z in x["features"]]
		test_suggestions = [z for x in test_instances_expanded for z in x["suggestions"]]

		test_suggestion_identifier = [a for b in test_suggestions for a in b]
		test_userdaf_identifier    = [ii for ii in range(len(test_history)) for a in range(len(test_suggestions[ii]))]

		test_train_instances_expanded = [my_train_from_test_splitter(x) for x in test_instances]
		test_labelled_history = [z for x in test_train_instances_expanded for z in x["features"]]
		test_labelled_suggestions = [z for x in test_train_instances_expanded for z in x["suggestions"]]
		test_labelled_response    = [z for x in test_train_instances_expanded for z in x["responses"]]

		Filter = [ii for ii in range(len(test_labelled_response)) if test_labelled_response[ii] in test_labelled_suggestions[ii]]

		test_labelled_history     = [test_labelled_history[ii] for ii in Filter]
		test_labelled_suggestions = [test_labelled_suggestions[ii] for ii in Filter]
		test_labelled_response    = [test_labelled_response[ii] for ii in Filter]
		##dict it up to feed it to feature extraction
		test_labelled = {'history': test_labelled_history,
						'suggestions': test_labelled_suggestions,
						'response' : test_labelled_response}

		count_in_instance = Counter(test_userdaf_identifier)
		self.test_query_count = [count_in_instance[i] for i in range(max(count_in_instance)+1)]

		test_X, _ , _ = Feature_extraction(test_history,test_suggestions,test_suggestion_identifier,test_userdaf_identifier,self.Binarisers,self.meta_info, self.filter_dict, self.item_popularity_dict, self.item_suggestion_dict, test_labelled=test_labelled)

		self.prob_predictions = list(self.model.predict(test_X))

		tmpdaf = pd.DataFrame({"index":test_userdaf_identifier,"prediction":self.prob_predictions,"suggestion":test_suggestion_identifier})

		predictions_per_user = [" ".join([str(x) for x in list(x[1].sort_values("prediction",ascending=False)["suggestion"])]) for x in tmpdaf.groupby("index")]

		pre_final = [x.tail(1)[["user_id","session_id","timestamp","step"]] for x in test_history]
		finaldaf = pd.concat(pre_final).reset_index(drop=True)

		finaldaf["item_recommendations"] = predictions_per_user

		self.final = finaldaf
		self.timer['predict'].append(time()-t)

		if out:
			return(finaldaf)

	def predictions_to_csv(self,name):
		self.final.to_csv("predictions/"+name,index=False)
	def record_test_preds(self, base_name):
		self.ensemble_preds(base_name+'test', self.prob_predictions, self.test_query_count)

	def record_valid_preds(self, base_name):
		preds = self.model.predict(self.watchlist[1].data)	
		self.ensemble_preds(base_name+'_valid', preds, self.validqc, self.watchlist[1].get_label())
	def create_fname(self):
		import datetime
		CT=datetime.datetime.now()
		k=[CT.day,CT.hour,CT.minute, self.num_feat]
		if self.valid_MRR is not None:
			k.append(np.round(self.valid_MRR,5))
		k = [str(x) for x in k]
		return '_'.join(k)
	def ensemble_preds(self, base_name ,preds, query_count, labels=None):
		if self.fname is None:
			self.fname=self.create_fname()
		fname = base_name+self.fname+'.csv'
		userdaf_identifier = [i for i, j in enumerate(query_count) for _ in range(j)]
		tmpdaf = pd.DataFrame({"index":userdaf_identifier,"prediction":preds})
		if labels is not None:
			tmpdaf['labels']=labels
		tmpdaf.to_csv('ensemble_preds/%s'%fname)
	def log_model(self, logname, newname=True):
		##Dump some random crap for our later use if needed
		original = sys.stdout
		if self.fname is None or newname==True: 
			self.fname=self.create_fname()
		sys.stdout = open('model/logs/'+logname+self.fname+'.log','w+')
		if self.valid_MRR is not None:
			print('The MRR was')
			print(self.valid_MRR)
		print('The parameters are')
		if self.optimal_rounds is not None: 
			print("Optimal number of boosting iterations:")
			print(self.optimal_rounds)
		for name in self.params.keys():
			print(name)
			print(self.params[name])
		print(self.timer) 
		print('feature Importances by split:')
		q = self.model.feature_importance()
		for i in q:
			print(i)
		print('feature Importances by gain')
		q=self.model.feature_importance(importance_type='gain')
		for i in q:
			print(i)
		sys.stdout=original

	def get_mask(self, logname, imp='split'):
		f = open('model/logs/'+logname, 'r')
		line_list = f.readlines()

		l1 = line_list.index('feature Importances by gain')
		l2 = line_list.index('feature Importances by split:')
		if imp=='split':
			start = l1+1
			end=l2
		if imp=='gain':
			start = l2
			end= len(line_list)
		importances= [float(line) for line in line_list[start:end]]



