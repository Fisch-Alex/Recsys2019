test  = [1,2,3,2,5,4,5,6,3,4,5,6,3]

test2  = [1,2,3,2,5,4,5,6,3,4,5,6,3,30]

all_ids  = set(test)
all_ids2 = set(test2) 

print(all_ids2-all_ids)

count_dict = {x : 0 for x in all_ids }

response = []

for ii in test: 
	response.append(count_dict[ii])
	count_dict[ii] += 1

print(response)