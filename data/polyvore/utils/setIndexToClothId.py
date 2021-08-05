"""
Convert Setid_index to Cloth id
"""

import os
import numpy as np
import json

polyvore_path = '..'
dataset_path = polyvore_path + '/dataset/'
jsons_path = polyvore_path + '/jsons/'
compat_file = jsons_path + 'fashion_compatibility_prediction.txt'
test_file = jsons_path + 'test_no_dup.json'

with open(test_file) as f:
    test_json_data = json.load(f)

map_id2their = {}
for fullSetData in test_json_data:
    outfit_ids = set()
    for item in fullSetData['items']:
        # get id from the image url
        _, id = item['image'].split('id=')
        id = int(id)
        outfit_indexid='{}_{}'.format(fullSetData['set_id'], item['index'])
        outfit_ids.add(outfit_indexid)
        map_id2their[outfit_indexid] = id

outfits = []
clothesid2set = []
allidset = {}

with open(compat_file) as f:
    for line in f:
        cols = line.rstrip().split(' ')
        items = cols[1:]
        outfits.append(items)
    i = 1
    for outfit in outfits:
        tmp=[]
        for cloth in outfit:         
            for v in map_id2their.keys():                
                if cloth == v:
                    tmp.append(map_id2their[v])
            allidset[i] = tmp
        i += 1

    print(allidset)
