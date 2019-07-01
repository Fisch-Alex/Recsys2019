import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.utils import sparsefuncs
from time import time
##Insanely ram heavy operation at the end
import gc
def User_summary(x):
	if len(x)>0:
	 	return [np.mean(x)+0.005, np.std(x)+0.005, stats.mode(x)[0][0]+0.005, np.min(x)+0.005, np.max(x)+0.005]
	else: 
		return [0,0,0,0,0]
def csr_summ(x):
	mean, var = sparsefuncs.mean_variance_axis(x, 0)
	min_val, max_val = sparsefuncs.min_max_axis(x, 0)
	return np.hstack([mean+0.005, var+0.005, min_val+0.005, max_val+0.005])
def Sequential_user_history(history ,item_dict, Number_of_Prices ):
	t = time()
	print('in seq user history')
	user_id = [x.user_id.iloc[-1] for x in history]
	time_stamp = [x.timestamp.iloc[-1] for x in history]
	sequential_df = item_dict['sequential_df']
	N = len(user_id)
	rel_freq_summary = [[0]*6*5 for i in range(N)]
	uniq_id = set(user_id)
	all_id = uniq_id.union(set(sequential_df['user_id']))

	# Use all_id to avoid future key errors 
	# We'll only ever actually use uid's in uniq_id
	user_ind_seq = {uid : [] for uid in all_id}
	user_ind_hist = {uid : [] for uid in uniq_id}
	up_to_forloop = time()-t
	print("First part of Sequential_user_history %s" % up_to_forloop)

	for i, key in sequential_df['user_id'].iteritems():
		user_ind_seq[key].append(i)
	for i, uid in enumerate(user_id):
		user_ind_hist[uid].append(i)

	up_to_forloop = time()-t
	print("making user inds %s" % up_to_forloop)
#### This is slower
#	rel_freq_summary=[User_summary(sequential_df.loc[(user_ind_mask[uid])&(sequential_df.loc[user_ind_mask[uid],'times']<ctime), 'true_rel_freq']) for uid,ctime in zip(user_id, time_stamp)]
	for uid in uniq_id: 
		user_hist = sequential_df.iloc[user_ind_seq[uid]]
		times = [time_stamp[i] for i in user_ind_hist[uid]]
		## reduce to past
		user_sequential_historys = [user_hist.loc[user_hist.times<ctime] for ctime in times]
		summaries = []
		for feat_name in ['true_rel_freq', 'rel_freq_rank', 'sugg_true_rel_freq', 'sugg_rel_freq_rank', 'clickout_perc_rank_true', 'clickout_perc_true']:
			summ = [User_summary(user_sequential_hist[feat_name]) for user_sequential_hist in user_sequential_historys]
			summaries.append(summ)
		for i, summary_lists in enumerate(user_ind_hist[uid]):
			summary_lists = [x for summary in summaries for x in summary[i]]
			rel_freq_summary[i] = summary_lists

	#print(rel_freq_summary[0])
	summary_full_len = [summary for summary, num in zip(rel_freq_summary, Number_of_Prices) for i in range(num) ]
	out1 = csr_matrix(summary_full_len)
	tt = time()-t
	print("Sequential_user_history: %s" % tt)
	return out1 

def User_History_Features(history, userdaf_identifier, cts_csrs,Number_Of_Prices, test_labelled=None, cat_csrs=[]):
	"""
		FUNCTION ASSUMES history HAS NO na clickouts in train case, and test_labelled['history'] in test case
		Also assumes Prices_Orded is generated from test_labelled['history'] in the test case

		cat_csrs must be tupled with a suitable binaraiser such that bin, csr will be returned while iterating.
	"""
	t2=time()

	timer = {}
	if test_labelled is None:
		## Obey rules, do everything sequentially for training data
		all_userid = set([x.user_id[0] for x in history])
		###Move into universe where all indices, are given in terms of a clickout history excluding na clickouts
		clickout_history = [x.iloc[-1] for x in history]
		clickout_history_inds = range(len(clickout_history))
		##Can use this list to go back to original indexing
		userid_history = [clickout_history[idx].user_id for idx in clickout_history_inds]
		## use ALL users so we don't run into key errrors
		user_dict={userid:[] for userid in all_userid}
		for history_idx, userid in enumerate(userid_history):
			user_dict[userid].append(history_idx)
		## This will be the indices (in our new index system)  of the past history of a user, 
		previous_history = [[ind for ind in user_dict[x.user_id[0]] if clickout_history_inds[ind]<state] for state, x in enumerate(history)]
		## Used to fetch rankings 
	if test_labelled is not None:
		## Use every piece of information that we can.
		## Don't do this sequentially
		## Obey rules, do everything sequentially for training data
		all_userid = set([x.user_id[0] for x in history+test_labelled['history']])
		###Move into universe where all indices, are given in terms of a clickout history excluding na clickouts
		clickout_history = [x.iloc[-1] for x in test_labelled['history']]
		##Can use this list to go back to original indexing
		clickout_history_inds = range(len(clickout_history))
		userid_history = [clickout_history[idx].user_id for idx in clickout_history_inds]

		## use ALL users so we don't run into key errrors
		user_dict={userid:[] for userid in all_userid}
		for history_idx, userid in enumerate(userid_history):
			user_dict[userid].append(history_idx)

		##
		## MAIN DIFFERENCE previous_history NOW CONTAINS ALL USER ITEMS
		##
		previous_history = [[ind for ind in user_dict[x.user_id[0]]] for state, x in enumerate(history)]
		## Used to fetch rankings 

	true_index = [x.impressions.split('|').index(x.reference) for x in clickout_history]
	true_index_history = [[true_index[ind] for ind in hist_list] for hist_list in previous_history]

	position_rank_list = [User_summary(x) for x in true_index_history]
	position_rank_csr = csr_matrix([position_rank_list[ind] for ind, num_prices in enumerate(Number_Of_Prices) for k in range(num_prices)])

#This is already in session csr
#	len_history = [len(x) for x in true_index_history]
#	len_history_csr = csr_matrix([[num] for num, num_prices in zip(len_history,Number_Of_Prices) for k in range(num_prices)])
	gc.collect()
	categorical_csr_stack=None
	for cat_csr, binariser in cat_csrs:	
		categorical_csr_stack =hstack((categorical_csr_stack,history_to_catsummcsr(cat_csr, clickout_history_inds, previous_history, true_index_history, Number_Of_Prices, binariser)))

	cts_csr_stack=None
	for cts_csr in cts_csrs:
		cts_csr_stack = hstack((cts_csr_stack, history_to_summcsr(cts_csr, clickout_history_inds, previous_history, true_index_history, Number_Of_Prices)))

	timer['totaluser'] = time()-t2
	print(timer)
	return hstack((position_rank_csr, cts_csr_stack,categorical_csr_stack))

def history_to_summcsr(list_of_interest, clickout_history_inds, previous_history, true_index_history, Number_of_Prices):
	interest_nona = [list_of_interest[i] for i in clickout_history_inds]
	interest_history = [[interest_nona[ind_hist][true_ind] for ind_hist, true_ind in zip(hist_list, true_ind)] for hist_list, true_ind in zip(previous_history, true_index_history)]
	interest_summ = [User_summary(x) for x in interest_history]
	interest_summ_csr = csr_matrix([x for x, num_prices in zip(interest_summ, Number_of_Prices) for k in range(num_prices)])
	return interest_summ_csr
def history_to_catsummcsr(list_of_interest, clickout_history_inds, previous_history, true_index_history, Number_of_Prices, binariser):
	interest_nona = [binariser(list_of_interest[i]) for i in clickout_history_inds]
	interest_history = [[interest_nona[ind_hist][true_ind,:] for ind_hist, true_ind in zip(hist_list, true_ind)] for hist_list, true_ind in zip(previous_history, true_index_history)]
	del interest_nona
	for x in interest_history:
		if len(x)>0:
			break
	default = csr_summ(vstack(x))*0

	interest_summ = [csr_summ(vstack(x)) if len(x)>0 else default for x in interest_history]
	del interest_history
	gc.collect()
	interest_summ_csr = csr_matrix([x for x, num_prices in zip(interest_summ, Number_of_Prices) for k in range(num_prices)])
	return interest_summ_csr
