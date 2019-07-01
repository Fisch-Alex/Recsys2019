import pandas as pd 
import lightgbm as lgb
import numpy as np

def my_train_splitter(x):
	indices     = [ii for ii in range(x.shape[0]) if x["action_type"][ii] == "clickout item"]
	features    = [x.iloc[0:(y+1),:] for y in indices]
	suggestions = [[int(y) for y in x["impressions"][z].split("|")] for z in indices]
	responses   = [int(x["reference"][y]) for y in indices]
	return({"features":features,"suggestions":suggestions,"responses": responses})

def my_train_from_test_splitter(x):
	indices     = [ii for ii in range(x.shape[0]) if x["action_type"][ii] == "clickout item" and not isinstance(x["reference"][ii], float)]
	features    = [x.iloc[0:(y+1),:] for y in indices]
	suggestions = [[int(y) for y in x["impressions"][z].split("|")] for z in indices]
	responses   = [int(x["reference"][y]) for y in indices]
	return({"features":features,"suggestions":suggestions,"responses": responses})

def my_test_splitter(x):
	indices     = [ii for ii in range(x.shape[0]) if x["action_type"][ii] == "clickout item" and isinstance(x["reference"][ii], float)]
	features    = [x.iloc[0:(y+1),:] for y in indices]
	suggestions = [[int(y) for y in x["impressions"][z].split("|")] for z in indices]
	return({"features":features,"suggestions":suggestions})

def split_idx(N,K):
	chunk_size = int(np.floor(N/K))
	split_inds = [i*chunk_size for i in range(0,K+1)]
	split_inds[-1]=N
	split_grouped_idx = [np.arange(split_inds[i],split_inds[i+1]) for i in range(len(split_inds)-1)]
	return split_grouped_idx
def cv_idx_gen(query_count,K, stratify):
	""" for use with LGBM CV needs to return:
	(generator or iterator of (train_idx, test_idx) tuples,

	Currently it's just a grouped K fold cross validation
	"""
	query_count = np.array(query_count)
	cum_sum_query = np.cumsum(query_count)

	if stratify is not None:
	#if stratify is the full data length
		if len(stratify)==query_count.sum():
			##-1 as we'll get the next group if not
			stratify = np.array(stratify)[cum_sum_query-1]
		ids = np.unique(stratify)
		id_idx = split_idx(len(ids),K)
		##Which ID's are in this group
		id_groups = [ids[idx] for idx in id_idx]
		all_instances = np.arange(len(stratify))
		split_grouped_idx = [all_instances[np.isin(stratify, id_group)] for id_group in id_groups]
	else:	
		N=len(query_count)
		split_grouped_idx = split_idx(N,K)
	split_groups = [query_count[idx] for idx in split_grouped_idx]
	split_groupcumsum = [cum_sum_query[idx] for idx in split_grouped_idx]
	
	dset_idx_fold = [np.hstack([np.arange(group_cumsum[i]-group[i], group_cumsum[i]) for i in range(len(group))]) for group, group_cumsum in zip(split_groups, split_groupcumsum)]
	##-1 as split_inds was generated on a shifted dataset (p insert(0,0))
	if K!=len(split_groups):
		assert ValueError("Changed value of K for CV to %s, Groups will not work, reduce number of folds"%K)
	test_train_id_iterator = [(np.hstack([x for j, x in enumerate(dset_idx_fold) if j!=i]), dset_idx_fold[i]) for i in range(K)]
	groups_iterator = [(np.hstack([x for j, x in enumerate(split_groups) if j!=i]), split_groups[i]) for i in range(K)]
	return test_train_id_iterator, groups_iterator

def tt_split_bad(ds ,query_count, K):
	"""This Does not work!!!! with ndcg for some reason
	Leaving here for reference"""
	idx_iterator, grp = cv_idx_gen(query_count,K)
	train = ds.subset(list(idx_iterator[K-1][0]))
	train.set_group(list(grp[K-1][0]))
	test = ds.subset(list(idx_iterator[K-1][1]))
	test.set_group(list(grp[K-1][1]))
	test.construct()
	train.construct()
	return(train, test)

def tt_split(ds, labels ,query_count, K, stratify=None, weights = None):
	labels = np.array(labels)
	idx_iterator, grp = cv_idx_gen(query_count,K, stratify)
	train = ds[(idx_iterator[K-1][0]),:]
	train_group = list(grp[K-1][0])
	test = ds[(idx_iterator[K-1][1]),:]
	test_group = list(grp[K-1][1])

	if weights is not None:
		train_weights = weights[(idx_iterator[K-1][0])]
		test_weights= weights[idx_iterator[K-1][1]]
		test_lgb = lgb.Dataset(test, label = labels[idx_iterator[K-1][1]],weight=test_weights ,group=test_group, free_raw_data=False)
		train_lgb = lgb.Dataset(train, label = labels[idx_iterator[K-1][0]],weight=train_weights ,group=train_group)
		return(train_lgb, test_lgb, train_weights, test_weights, test_group)

	else:
		test_lgb = lgb.Dataset(test, label = labels[idx_iterator[K-1][1]], group=test_group, free_raw_data=False)
		train_lgb = lgb.Dataset(train, label = labels[idx_iterator[K-1][0]], group=train_group)
	return(train_lgb, test_lgb, test_group)
