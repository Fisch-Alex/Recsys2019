import numpy as np
import pandas as pd 
from Feature_extraction import binary_arr
from collections import Counter
def extract_counter(inp_file):
	data = pd.read_csv(inp_file)
	red = data[data['action_type']=='filter selection']
	current_filters = data['current_filters'][data['current_filters'].notna()]
	current_filters_lists = current_filters.apply(lambda x: x.split('|')) 
	current_filter_list = [j for i in current_filters_lists for j in i]

	C_current = Counter(current_filter_list)
	C_toggle=Counter(red['reference'])

	return C_current+C_toggle
def gen_filter_dict(train_file, test_file, thresh=60):

	C_train=extract_counter(train_file)
	C_test=extract_counter(test_file)

	all_filters = set([c for c in C_test]+[c for c in C_train])
	filter_interest = [c for c in C_test if C_test[c]>thresh if C_train[c]>thresh]
	n_filter = len(filter_interest)

	filter_dict_interest={filter_interest[i]: binary_arr(i,n_filter) for i in range(n_filter)}
	zero_arr = binary_arr(n_filter+1, n_filter)
	filter_dict = {c:zero_arr for c in all_filters}
	filter_dict.update(filter_dict_interest) 
	filter_dict['zero'] = zero_arr

	return(filter_dict)
