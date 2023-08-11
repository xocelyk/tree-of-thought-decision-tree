# 5-shot
standard_prompt = '''
Choose a feature and value on which to split the data.
###
{feature}: {feature_value_list}
'''

def standard_prompt(feat_val_map: dict, prev_splits: list) -> str:
    return '''
    Choose a feature and value on which to split the data.
    ###
    Input:
    Longitude: -121.27
    Latitude: 38.69
    Housing Median Age: 16
    Total Rooms: 3389
    Population: 1674
    Number Households: 568
    Median Income: 44489
    Previous Splits: Median Income < 50000, Population < 2000
    Answer: Median Income < 50000, Population < 2000, Total Rooms < 4000
    ###
    Input:
    Longitude: -118.39
    Latitude: 33.96
    Housing Median Age: 45
    Total Rooms: 1436
    Total Bedrooms: 374
    Population: 662
    Number Households: 292
    Median Income: 36250
    Previous Splits: Median Income > 35000, Housing Median Age > 40, Population < 1000
    Answer: Median Income > 35000, Housing Median Age > 40, Population < 1000, Total Rooms > 1000
    ###
    Input:
    {}'''.format('\n'.join([f'{k}: {v}' for k, v in feat_val_map.items()]) + '\nPrevious Splits: ' + ', '.join(prev_splits))

