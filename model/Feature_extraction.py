import pandas as pd 
import numpy as np
from time import time
from scipy import stats
from scipy.sparse import csr_matrix, hstack
import os
from sklearn import preprocessing
from multiprocessing import Pool
from Feature_functions import Sequential_user_history, User_History_Features
import gc


def Popularity_train(history,suggestions,userdaf_identifier,item_dict,item_dict2):

	all_ids = set([x for y in suggestions for x in y])
	item_dict.update({x : 0 for x in all_ids })
	item_dict2.update({x : 0 for x in all_ids })

	answer = []
	totals = []
	suggestion_answer = []
	suggestion_totals = []

	times = [list(x['timestamp'])[-1] for x in history] 
	truth = [int(list(x['reference'])[-1]) for x in history] 
	user_id = [list(x['user_id'])[-1] for x in history] 
	tmp_daf = pd.DataFrame({'times': times, 'truth': truth, 'suggestions': suggestions, 'user_id': user_id})
	tmp_daf.sort_values("times", inplace = True)

	ordered_truth = list(tmp_daf['truth'])
	ordered_suggestions = list(tmp_daf['suggestions'])

	tmp_daf['truth_ind'] = tmp_daf.apply(lambda x: x.suggestions.index(x.truth), axis=1)

	for ii in range(len(ordered_truth)):
		answer.append([item_dict[x] for x in ordered_suggestions[ii]])
		totals.append([item_dict["total"] for x in ordered_suggestions[ii]])
		item_dict[ordered_truth[ii]] += 1
		item_dict['total'] += 1
		current_sugg = []
		for sugg in ordered_suggestions[ii]:
			item_dict2['total'] += 1
			item_dict2[sugg   ] += 1
			current_sugg.append(item_dict2[sugg])
		suggestion_answer.append(current_sugg)
		suggestion_totals.append([item_dict2["total"] for x in ordered_suggestions[ii]])

	### Store all history in frequencies
	### Using arrays as we can easily do component wise division
	tmp_daf['frequencies'] = [np.array(x) for x in answer]
	tmp_daf['total']       = [np.array(x) for x in totals]
	tmp_daf['subtotals']   = tmp_daf.loc[:,'frequencies'].apply(np.sum)+1
	tmp_daf['rel_freq']    = tmp_daf.apply(lambda x: x.frequencies/x.subtotals, axis=1)
	
	tmp_daf['rel_freq_rank'] = tmp_daf.apply(lambda x: x.rel_freq.argsort().argsort()[x.truth_ind], axis=1)
	tmp_daf['true_rel_freq'] = tmp_daf.apply(lambda x: x.rel_freq[x.truth_ind], axis=1)

	# Note that no +1 is required!
	tmp_daf['sugg_frequencies'] = [np.array(x) for x in suggestion_answer]
	tmp_daf['sugg_total']       = [np.array(x) for x in suggestion_totals]
	tmp_daf['sugg_subtotals']   = tmp_daf.loc[:,'sugg_frequencies'].apply(np.sum)
	tmp_daf['sugg_rel_freq']    = tmp_daf.apply(lambda x: x.sugg_frequencies/x.sugg_subtotals, axis=1)

	tmp_daf['sugg_rel_freq_rank'] = tmp_daf.apply(lambda x: x.sugg_rel_freq.argsort().argsort()[x.truth_ind], axis=1)
	tmp_daf['sugg_true_rel_freq'] = tmp_daf.apply(lambda x: x.sugg_rel_freq[x.truth_ind], axis=1)

	tmp_daf['clickout_perc'] = tmp_daf.apply(lambda x: x.frequencies/x.sugg_frequencies, axis=1)
	tmp_daf['clickout_perc_rank'] = tmp_daf.apply(lambda x: x.clickout_perc.argsort().argsort(), axis=1)

	tmp_daf['clickout_perc_true'] = tmp_daf.apply(lambda x: x.clickout_perc[x.truth_ind], axis=1)
	tmp_daf['clickout_perc_rank_true'] = tmp_daf.apply(lambda x: x.clickout_perc_rank[x.truth_ind], axis=1)
	tmp_daf.sort_index(inplace = True)

	output1 = [x for y in list(tmp_daf['frequencies']) for x in y]
	output2 = [x for y in list(tmp_daf['total']) for x in y]
	output3 = [z for y in list(tmp_daf['frequencies'])  for z in [np.sum(y)+1]*len(y) ]

	output4 = [x for y in list(tmp_daf['sugg_frequencies']) for x in y]
	output5 = [x for y in list(tmp_daf['sugg_total']) for x in y]
	output6 = [z for y in list(tmp_daf['sugg_frequencies'])  for z in [np.sum(y)]*len(y) ]


	output7 = csr_matrix([[z] for y in list(tmp_daf['clickout_perc'])  for z in y ])
	###Just store what we need at testing time.
	### .loc so it returns a copy
	item_dict['sequential_df'] = tmp_daf.loc[:,['user_id','times','true_rel_freq','rel_freq_rank','sugg_true_rel_freq','sugg_rel_freq_rank', 'clickout_perc_true', 'clickout_perc_rank_true']]
	return output1, output2, output3, output4, output5, output6, output7 ,item_dict, item_dict2



def binary_arr(k=9, n=8):
	l = [1 if k==i else 0 for i in range(n)]
	return(np.array(l).reshape(1,n))


sort_order_encoder = {
		'interaction sort button' : binary_arr(0),
		'price and recommended': binary_arr(1),
		'price only':binary_arr(2),
       'distance only':binary_arr(3),
	   'rating only':binary_arr(4),
	   'rating and recommended':binary_arr(5),
       'distance and recommended':binary_arr(6),
	   'our recommendations':binary_arr(7)
		}

	

def Popularity_test(history,suggestions,userdaf_identifier,item_dict,item_dict2):

	all_ids = set([x for y in suggestions for x in y])
	item_dict.update({x : 0 for x in (all_ids - set(item_dict.keys())) })
	item_dict2.update({x : 0 for x in (all_ids - set(item_dict2.keys())) })

	suggestion_answer = []
	suggestion_totals = []

	times = [list(x['timestamp'])[-1] for x in history] 
	user_id = [list(x['user_id'])[-1] for x in history] 
	tmp_daf = pd.DataFrame({'times': times, 'suggestions': suggestions, 'user_id': user_id})
	tmp_daf.sort_values("times", inplace = True)

	ordered_suggestions = list(tmp_daf['suggestions'])

	# guess what the .000000001 is there for! 
	output4_not_incremented = [ item_dict2[x]+0.0000001 for y in suggestions for x in y ]

	for ii in range(len(ordered_suggestions)):
		current_sugg = []
		for sugg in ordered_suggestions[ii]:
			item_dict2['total'] += 1
			item_dict2[sugg   ] += 1
			current_sugg.append(item_dict2[sugg])
		suggestion_answer.append(current_sugg)
		suggestion_totals.append([item_dict2["total"] for x in ordered_suggestions[ii]])

	pre_output1 = [ [item_dict[x]  for x in y ] for y in suggestions ]
	
	tmp_daf['sugg_frequencies'] = [np.array(x) for x in suggestion_answer]
	tmp_daf['sugg_total']       = [np.array(x) for x in suggestion_totals]
	tmp_daf['sugg_subtotals']   = tmp_daf.loc[:,'sugg_frequencies'].apply(np.sum)
	tmp_daf['sugg_rel_freq']    = tmp_daf.apply(lambda x: x.sugg_frequencies/x.sugg_subtotals, axis=1)

	tmp_daf['clickout_perc']    = tmp_daf.apply(lambda x: x.sugg_frequencies/x.sugg_subtotals, axis=1)

	tmp_daf.sort_index(inplace = True)

	output1 = [ x for y in pre_output1 for x in y ]

	output4 = [x for y in list(tmp_daf['sugg_frequencies']) for x in y]
	output5 = [x for y in list(tmp_daf['sugg_total']) for x in y]
	output6 = [z for y in list(tmp_daf['sugg_frequencies'])  for z in [np.sum(y)]*len(y) ]
	clickout_perc = [np.array(x)/np.array(y) for x, y in zip( pre_output1, list(tmp_daf['sugg_frequencies']))]
	clickout_perc_ranked = [x.argsort().argsort() for x in clickout_perc]
	output7 = csr_matrix([[output1[ii]/output4_not_incremented[ii]] for ii in range(len(output4_not_incremented))])
	return output1, [ item_dict["total"] for y in suggestions for x in y ], [z for y in pre_output1 for z in [np.sum(y)+1]*len(y) ], output4, output5, output6, output7

def Feature_extraction(history,suggestions,suggestion_identifier,userdaf_identifier,Binarisers,meta_info,filter_dict,item_dict,item_dict2,response = None ,test_labelled=None, ncores=4, min_pop=50, cut_off=15):

	meta_info_dict = meta_info.set_index("item_id").to_dict('index')
	Country_daf          = Binarisers[0].transform([history[ii]["platform"][0] for ii in userdaf_identifier])
	Property_type_matrix = Binarisers[1].transform([[meta_info_dict[suggestion]["type of property"] ] for suggestion in suggestion_identifier])
	Device_daf           = Binarisers[2].transform([history[ii]["device"][0] for ii in userdaf_identifier])
	properties_reduced_daf = Binarisers[5].transform([meta_info_dict[suggestion]["properties_reduced"] for suggestion in suggestion_identifier])

	if response is None:
		frequencies, totals, sub_totals, sugg_frequencies, sugg_totals, sugg_sub_totals, pop_fraction  = Popularity_test(history,suggestions,userdaf_identifier,item_dict,item_dict2)
	else: 
		frequencies, totals, sub_totals, sugg_frequencies, sugg_totals, sugg_sub_totals, pop_fraction,item_dict, item_dict2 = Popularity_train(history,suggestions,userdaf_identifier,item_dict,item_dict2)
	
	relative_frequencies = csr_matrix([[frequencies[ii]/sub_totals[ii]] for ii in range(len(userdaf_identifier))])
	
	relative_totals = csr_matrix([[sub_totals[ii]/(totals[ii]+1)] for ii in range(len(userdaf_identifier))])

	relative_sugg_frequencies = csr_matrix([[sugg_frequencies[ii]/sugg_sub_totals[ii]] for ii in range(len(userdaf_identifier))])

	relative_sugg_totals = csr_matrix([[sugg_sub_totals[ii]/(sugg_totals[ii])] for ii in range(len(userdaf_identifier))])
	
	###

	##### Features to do with Prices
	country_dict = Binarisers[-1]
	citycountry = [x['city'].iloc[-1] for x in history]
	Country = [x.split(', ')[1] for x in citycountry]
	#city_df = pd.DataFrame({'Country':[x[1] for x in citycountrysplit], 'City': [x[0] for x in citycountrysplit]})

	#for country in set(city_df.loc[:,'Country']):
#		if country not in country_dict:
#			country_dict[country]=set()


	City_to_daf        = Binarisers[3].transform([[citycountry[ii]] for ii in userdaf_identifier])
	Country_to_daf        = Binarisers[4].transform([[Country[ii]] for ii in userdaf_identifier])

	Prices =  [[int(y) for y in list(x["prices"])[-1].split("|")] for x in history]
	Prices_Orded = [list(np.array(x).argsort().argsort()) for x in Prices]
	Number_Of_Prices = [len(x) for x in Prices]

	Number_Of_Prices_Expanded = csr_matrix([[Number_Of_Prices[suggestion]] for suggestion in userdaf_identifier])

	#The plus .5 doesn't matter for this feature but for the relative one!
	Prices_Orded_Expanded = csr_matrix([[x+0.5] for y in Prices_Orded for x in y])

	Mean_Prices = [np.mean(x) for x in Prices] 

	Mean_Prices_Expanded = csr_matrix([[Mean_Prices[suggestion]] for suggestion in userdaf_identifier])

	List_of_Prices = csr_matrix([[x] for y in Prices for x in y])

	Relative_Price =  List_of_Prices/Mean_Prices_Expanded
	Relative_Rank  =  (Prices_Orded_Expanded)/Number_Of_Prices_Expanded


	
	##Censor for now, shouldn't be neccesary when done sequentially


	"""
		Currently not using frac features in our model will be worth a try when sequential popularity added in 
	"""

	Position_vector     = csr_matrix([[ii] for element in suggestions for ii in range(len(element))])

	def str_to_list_of_props(string, meta_info_dict, suggestions_in):
		return [[meta_info_dict[suggestion][string] for suggestion in suggestion_list] for suggestion_list in suggestions_in]
	def sugg_to_listoflists(str_iter, meta_info_dict, suggestions_in):
		return [str_to_list_of_props(string, meta_info_dict, suggestions_in) for string in str_iter]
	
	str_iter = ['Star', 'Wifi', 'Pets', 'Family','TV', 'Jacuzzi', 'Sauna', 'Swimming Pool', 'Restaurant', 'suggestion_frequency']
	if test_labelled is None:
		property_lists = [[meta_info_dict[suggestion]["type of property"] for suggestion in suggestion_list] for suggestion_list in suggestions]
		properties_reduced_lists = [[meta_info_dict[suggestion]["properties_reduced"] for suggestion in suggestion_list] for suggestion_list in suggestions]

		meta_ll = sugg_to_listoflists(str_iter, meta_info_dict, suggestions)
	if test_labelled is not None:
		Prices =  [[int(y) for y in list(x["prices"])[-1].split("|")] for x in test_labelled['history']]

		property_lists = [[meta_info_dict[suggestion]["type of property"] for suggestion in suggestion_list] for suggestion_list in test_labelled['suggestions']]
		properties_reduced_lists = [[meta_info_dict[suggestion]["properties_reduced"] for suggestion in suggestion_list] for suggestion_list in test_labelled['suggestions']]

		meta_ll = sugg_to_listoflists(str_iter, meta_info_dict, test_labelled['suggestions'])
		Prices_Orded = [list(np.array(x).argsort().argsort()) for x in Prices]

	gc.collect()
	print(meta_ll[0][1])
	meta_ll.append(Prices)
	meta_ll.append(Prices_Orded)

	## The version with cat_csrs is better but has caused issues in the past
	User_Features = User_History_Features(history, userdaf_identifier, meta_ll,Number_Of_Prices, test_labelled, cat_csrs= [(property_lists,Binarisers[1].transform)])
	#User_Features = User_History_Features(history, userdaf_identifier, meta_ll,Number_Of_Prices, test_labelled, cat_csrs= [])
	
	Sequential_User_Features = Sequential_user_history(history, item_dict, Number_Of_Prices)
	#if response is not None:
	#	Number_Of_Clicks = csr_matrix([[meta_info_dict[suggestion_identifier[ii]]["click_frequency"] - response[ii] ] for ii in range(len(suggestion_identifier))] )
	#else:
	#	Number_Of_Clicks = csr_matrix ([[meta_info_dict[suggestion]["click_frequency"] ] for suggestion in suggestion_identifier] )
	gc.collect()
	meta_properties = [[meta_info_dict[suggestion][string] for string in str_iter] for suggestion in suggestion_identifier]


	SessionCSR = Session_features(suggestions, userdaf_identifier, suggestion_identifier, history, filter_dict)
	HistoryCSR = Session_features(suggestions, userdaf_identifier, suggestion_identifier, history, filter_dict, session=False)
	
	Number_Of_Suggestions = csr_matrix([[meta_info_dict[suggestion]["suggestion_frequency"] ] for suggestion in suggestion_identifier])
	# Let's remove the dodgy features for now!!!
	#Featuredaf = hstack((Country_daf,Is_Contained,Number_Contained,Amount_of_info,Number_Of_Clicks,Number_Of_Suggestions,Has_wifi,Stars,Property_type_matrix)).tocsr()

	Featuredaf = hstack((Position_vector, pop_fraction, relative_frequencies, relative_totals, relative_sugg_frequencies, Number_Of_Suggestions,  relative_sugg_totals, Device_daf,Relative_Price,Relative_Rank,Country_daf, Country_to_daf, City_to_daf, properties_reduced_daf,meta_properties,Property_type_matrix,List_of_Prices,Prices_Orded_Expanded, Number_Of_Prices_Expanded,Mean_Prices_Expanded, SessionCSR,HistoryCSR, User_Features,  Sequential_User_Features)).tocsr()
	print(Featuredaf.shape)

	return Featuredaf, item_dict, item_dict2

def Session_features(suggestions, userdaf_identifier, suggestion_identifier, history, filter_dict,session=True, meta_info_dict=None):
	t_total=time()
	if session:
		history= [x.loc[x['session_id']==x['session_id'].iloc[-1]] for x in history]
	timer={}
	t=time()

	References_by_user  = [list(x["reference"][:-1]) for x in history]
	Suggestions_by_user = [list(x["impressions"][:-1]) for x in history]

	References_by_user_no_nas = [  [ y for y in x if not isinstance(y, float) ]  for x in  References_by_user]

	Integer_References_by_user = [  [ y for y in x if y.isdigit() ]  for x in  References_by_user_no_nas]
	Actions_by_user = [list(x["action_type"][:-1]) for x in history]
	timer['reduce_hist'] = time()-t

	t=time()
	References_clickouts  = [[ref for action, ref in zip(actions, references) if action=='clickout item' if not isinstance(ref, float)] for actions, references in zip(Actions_by_user,References_by_user)]
	Suggestion_history    = [[Sug for action, Sug in zip(actions, Suggestions) if action=='clickout item' if not isinstance(Sug, float)] for actions, Suggestions in zip(Actions_by_user,Suggestions_by_user)]
	Suggestion_history    = [[int(a) for b in x for a in b.split("|")] for x in Suggestion_history]
	References_itemimage  = [[ref for action, ref in zip(actions, references) if action=='interaction item image'] for actions, references in zip(Actions_by_user,References_by_user)]
	References_iteminfo   = [[ref for action, ref in zip(actions, references) if action=='interaction item info'] for actions, references in zip(Actions_by_user,References_by_user)]
	References_itemsearch = [[ref for action, ref in zip(actions, references) if action=='search for item'] for actions, references in zip(Actions_by_user,References_by_user)]
	References_itemdeals  = [[ref for action, ref in zip(actions, references) if action=='interaction item deals'] for actions, references in zip(Actions_by_user,References_by_user)]
	References_itemrating = [[ref for action, ref in zip(actions, references) if action=='interaction item rating'] for actions, references in zip(Actions_by_user,References_by_user)]
	timer['ref_type']     = time()-t
	t=time()
	List_of_item_sets      = [set(x) for x in Integer_References_by_user]
	List_of_item_lengths   = [len(x) for x in Integer_References_by_user]

	List_of_sets      = [set(x) for x in References_by_user]
	List_of_lengths   = [len(x) for x in References_by_user]

	List_of_clickout_sets = [set(x) for x in References_clickouts]
	List_of_clickout_lens = [len(x) for x in References_clickouts]
	
	List_of_info_lens = [len(x) for x in References_iteminfo]
	
	List_of_image_lens  = [len(x) for x in References_itemimage]

	List_of_search_lens = [len(x) for x in References_itemsearch]

	List_of_deals_lens  = [len(x) for x in References_itemdeals]

	List_of_rating_lens = [len(x) for x in References_itemrating]

	clickout_interaction_disjoint = [len(set_interact-set_click) for set_interact, set_click in zip(List_of_item_sets, List_of_clickout_sets)]
	Amount_of_overlap = [len(List_of_item_sets[ii].intersection(set([str(x) for x in suggestions[ii]]))) for ii in range(len(List_of_item_sets)) ]
	timer['makinglists']=time()-t
	
	t=time()
	Is_Contained            = csr_matrix([[1.0] if str(suggestion_identifier[ii]) in List_of_sets[userdaf_identifier[ii]] else [0.0] for ii in range(len(suggestion_identifier))])
	Number_Contained        = csr_matrix([[List_of_lengths[userdaf_identifier[ii]]] for ii in range(len(suggestion_identifier))])
	Number_item_Contained   = csr_matrix([[List_of_item_lengths[userdaf_identifier[ii]]] for ii in range(len(suggestion_identifier))])
	Amount_of_info          = csr_matrix([[Amount_of_overlap[userdaf_identifier[ii]]] for ii in range(len(suggestion_identifier))])

	### Actual Features: 
	Number_of_interactions = csr_matrix([ [Integer_References_by_user[userdaf_identifier[ii]].count(str(suggestion_identifier[ii]))] for ii in range(len(suggestion_identifier))])
	timer['first5csr']= time()-t
	t=time()
	previous_clickout_suggestion   = csr_matrix([[References_clickouts[hist_ind].count(str(suggestion))] for hist_ind, suggestion in zip(userdaf_identifier, suggestion_identifier)])
	previous_suggestion_suggestion = csr_matrix([[Suggestion_history[hist_ind].count(int(suggestion))] for hist_ind, suggestion in zip(userdaf_identifier, suggestion_identifier)])


	previous_user_numclickout = csr_matrix([[List_of_clickout_lens[hist_ind]] for hist_ind, suggestion in zip(userdaf_identifier, suggestion_identifier)])
	previous_user_numinfo = csr_matrix([[List_of_info_lens[hist_ind]] for hist_ind, suggestion in zip(userdaf_identifier, suggestion_identifier)])
	previous_user_numimage = csr_matrix([[List_of_image_lens[hist_ind]] for hist_ind, suggestion in zip(userdaf_identifier, suggestion_identifier)])
	previous_user_numsearch = csr_matrix([[List_of_search_lens[hist_ind]] for hist_ind, suggestion in zip(userdaf_identifier, suggestion_identifier)])
	previous_user_numdeals = csr_matrix([[List_of_deals_lens[hist_ind]] for hist_ind, suggestion in zip(userdaf_identifier, suggestion_identifier)])
	previous_user_numrating = csr_matrix([[List_of_rating_lens[hist_ind]] for hist_ind, suggestion in zip(userdaf_identifier, suggestion_identifier)])

	Look_at_no_clickout = csr_matrix([[clickout_interaction_disjoint[ii]] for ii in userdaf_identifier])

	timer['prev_action_csr']=time()-t
	t=time()
	Last_interaction      = [ "DUMMY" if len(x) == 0 else x[-1] for x in References_by_user]
	Last_item_interaction = [ "DUMMY" if len(x) == 0 else x[-1] for x in Integer_References_by_user]

	Is_Last_interaction  = csr_matrix([[1.0] if str(suggestion_identifier[ii]) == Last_interaction[userdaf_identifier[ii]] else [0.0] for ii in range(len(suggestion_identifier))])
	Is_Last_item_interaction  = csr_matrix([[1.0] if str(suggestion_identifier[ii]) == Last_item_interaction[userdaf_identifier[ii]] else [0.0] for ii in range(len(suggestion_identifier))])
	
	
	
	timer['islast'] = time()-t

	### Sort order 
	n_filter = len(sort_order_encoder)
	zeroarr = binary_arr(n_filter+1, n_filter) 

	References_sortorder = [[ref for action, ref in zip(actions, references) if action=='change sort order'] for actions, references in zip(Actions_by_user,References_by_user)]
	last_sort = [sort_order_encoder[refs[-1]] if len(refs)>0 else zeroarr for refs in References_sortorder]
	all_sorts = [np.vstack([sort_order_encode[ref] for ref in refs]) if len(refs)>0 else zeroarr for refs in References_sortorder]
	sum_all_sorts = [arr.sum(axis=0) for arr in all_sorts]
	sum_all_sorts_csr = csr_matrix([sum_all_sorts[hist_ind].tolist() for hist_ind in userdaf_identifier])
	last_sorts_csr = csr_matrix([last_sort[hist_ind].flatten().tolist() for hist_ind in userdaf_identifier])
	
	### Filters
	zeroarr_key = 'zero'
	zeroarr = filter_dict[zeroarr_key]

	References_filter_toggle = [[ref for action, ref in zip(actions, references) if action=='filter section'] for actions, references in zip(Actions_by_user,References_by_user)]
	last_toggle = [filter_dict[refs[-1]] if len(refs)>0 else zeroarr for refs in References_filter_toggle]
	all_toggles = [np.vstack([filter_dict[ref] for ref in refs]) if len(refs)>0 else zeroarr for refs in References_filter_toggle]
	sum_all_toggle = [arr.sum(axis=0) for arr in all_toggles]

	sum_all_toggle_csr = csr_matrix([sum_all_toggle[hist_ind].tolist() for hist_ind in userdaf_identifier])
	last_toggle_csr = csr_matrix([last_toggle[hist_ind].flatten().tolist() for hist_ind in userdaf_identifier])
	

	current_filters = [[z.split('|') for z in x['current_filters'] if not isinstance(z, float)] for x in history]
	last_filter_list = [filters_list[-1] if len(filters_list)>0 else ['zero'] for filters_list in current_filters]
	last_filter_encoded = [np.vstack([filter_dict[filter_el] for filter_el in filter_list]).sum(axis=0) for filter_list in last_filter_list]

	last_filter_csr = csr_matrix([last_filter_encoded[hist_ind].tolist() for hist_ind in userdaf_identifier])

	timer['full_session'] = time()-t_total
	print("Session_features time: %s "%timer['full_session'])

	features = hstack((Is_Contained, Number_Contained, Number_item_Contained, Amount_of_info, Number_of_interactions, previous_suggestion_suggestion, previous_clickout_suggestion, previous_user_numclickout, previous_user_numinfo, previous_user_numimage, previous_user_numsearch, previous_user_numdeals, previous_user_numrating, Look_at_no_clickout, Is_Last_interaction, Is_Last_item_interaction,sum_all_sorts_csr, last_sorts_csr, sum_all_toggle_csr, last_filter_csr, last_toggle_csr))
	### Note if you want to add things just for the non session case, just if loop it here and add a hstack:
#	if not session:
	return features 

