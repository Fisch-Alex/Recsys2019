##just made a script called user_args containing anything user specific to make things slightly easier
from item_feature_functions import extract_properties_meta, get_item_frequency
from collections import OrderedDict
import pandas as pd
import os
my_dir = os.environ['Trivago']
os.chdir(my_dir)

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv('data/test.csv')
metadata = pd.read_csv("data/item_metadata.csv")
## feat_map to group similar properties into columns
feat_map ={'type of property': OrderedDict({
                                'BB' : ['Bed & Breakfast'],
                               'House': ['House / Apartment', 'Serviced Apartment', 'Self Catering'], ##unsure about self catering
                               'Hostel': ['Hostel'],
                               'Hotel': ['Hotel', 'Terrace (Hotel)', 'Business Hotel', 'Country Hotel', 'Convention Hotel'],
                               'Prem_Hotel' : ['Luxury Hotel'] 
                               }),
            ##Stars need to be ordered, 
            #this allows us to check from 1 star before from 2 star, etc
           #Ordering will give the highest rating it lists
            'Star':OrderedDict({1: ['1 Star', 'From 1 Stars'],
                    2: ['2 Star', 'From 2 Stars'],
                    3: ['3 Star', 'From 3 Stars'],
                    4: ['4 Star', 'From 4 Stars'],
                    5: ['5 Star']}),
            'Pets': {1:['Pet Friendly']},
            'Family': {1:['Family_Friendly']},
            'TV' : {1:['Satellite TV', 'Flatscreen TV', 'Cable TV', 'Television']},
            'Restaurant' : {1:['Restaurant']},
            'Swimming Pool' : {1:['Swimming Pool (Indoor)', 'Swimming Pool (Combined Filter)', 'Swimming Pool (Outdoor)', 'Swimming Pool (Bar)']},
            'Jacuzzi' : {1:['Jacuzzi (Hotel)']},
            'Sauna' : {1:['Sauna']},
            'Wifi': {1: ['WiFi', 'Free WiFi (Combined)', 'WiFi (Public Areas)', 'Free WiFi (Public Areas)', 'Telephone','Free WiFi (Rooms)', 'WiFi (Rooms)']}
}

## Include anything in drop_feat which is likely to be useless.
drop_feat = set(['Shower'])
print('Counting occurances in train data')
count_train = get_item_frequency(train_data)
print('Counting occurances in test data')
count_test = get_item_frequency(test_data)
print('extract properties starting')
clean_meta = extract_properties_meta(metadata, feat_map, drop_feat)
s='|'
clean_meta.loc[:,'properties_reduced'] = clean_meta['properties_reduced'].apply(lambda x: s.join(x))
updated_meta = count_train.merge(clean_meta, on ="item_id", how='outer')
updated_meta = updated_meta.merge(count_test, on ="item_id", how='outer')

count_columns = ['suggestion_frequency_x', 'suggestion_frequency_y','click_frequency_x', 'click_frequency_y']
updated_meta.loc[:,count_columns]= updated_meta.loc[:,count_columns].fillna(0)

## Two columns for train and test will not be added, they'll be named *_x, *_y, so add these
updated_meta['click_frequency']= updated_meta['click_frequency_x']+updated_meta['click_frequency_y']
updated_meta['suggestion_frequency']= updated_meta['suggestion_frequency_x']+updated_meta['suggestion_frequency_y']

## Drop *_x, *_y columns, could undo this if we ever need test frequencies
updated_meta.drop(updated_meta.filter(regex='frequency_').columns, axis=1,inplace=True)

## fill categorical columns with strings rather than Na. prevents future errors
updated_meta.fillna({'type of property':'missing', 'properties_reduced':'none'}, inplace=True)
updated_meta.to_csv("data/metadata_updated.csv",index=False)
