import numpy as np
import pandas as pd
import os
from collections import Counter


def extract_properties_meta(metadata, feat_map, drop_feat):
    """
    input: 
        metadata which is read in as a pd dataframe 
        feat_map structured at dict[dict[ list]]
        drop_feat, a list/set of things to drop
    outputs: 
        updated metadata with features split into columns, residual properties left in 'resid column'
        
    Note: Not efficient but should only be ran rarely
    """
    n = metadata.shape[0]
    out_dict={}
    propertylists = metadata.apply(lambda x: x.properties.split('|'), axis=1)
    for feat_key, _ in feat_map.items():
        out_dict[feat_key]=[0]*n
    out_dict['item_id']=metadata.loc[:,'item_id']
    out_dict['properties_reduced'] = propertylists
    for i in range(n):
        row = metadata.iloc[i]
        for feat_key, feat_dict in feat_map.items():
            for val, level_list in feat_dict.items():
                if not set(level_list).isdisjoint(propertylists[i]):
                    out_dict[feat_key][i] = val
                    out_dict['properties_reduced'][i] = [x for x in out_dict['properties_reduced'][i] if x not in level_list and x not in drop_feat]
    ret = pd.DataFrame(out_dict)
    ##change all 0s to NA
    ##I've check no item_id is NA
    ret.fillna(0)
    return(ret)

def get_item_frequency(train_data):
    """
    outputs a dataframe similar to metadata with counts of how many click outs and how many times it has been suggested
    """
    clicked_on = train_data.loc[train_data["action_type"] == "clickout item",:]["reference"]
    suggested = train_data.loc[train_data["action_type"] == "clickout item",:]["impressions"]
    suggested = [y for x in suggested for y in x.split("|")]
    suggestion_frequencies = Counter(suggested) 
    click_frequencies      = Counter(clicked_on) 
    info_daf={}
    #ids = list(info_daf["item_id"])
    info_daf['item_id'] = [int(x) for x in suggestion_frequencies]

    info_daf["click_frequency"]      = [click_frequencies[str(x)] for x in info_daf['item_id']]
    info_daf["suggestion_frequency"] = [suggestion_frequencies[str(x)] for x in info_daf['item_id']]
    info_daf=pd.DataFrame(info_daf)
    return(info_daf)
