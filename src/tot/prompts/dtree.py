import numpy as np

STRINGS = {
"SYSTEM_CONTENT_1": "You are Charlie, an AI housing price expert. You are studying California median housing price data. Your job is to predict whether or not the median house value for households within a block is greater than $200,000. After assessing your predictions, you will be asked to provide a hypothesis between the features in the housing data and the median house value being greater than $200,000. The goal is to generate a hypothesis that will be useful to a future agent performing the same prediction task.",
"USER_CONTENT_1": "Your answer must end with either: '(A) The median house value for houses in this block is greater than $200,000.' or '(B) The median house value for houses in this block is less than or equal to $200,000.' Do you understand?",
"ASSISTANT_CONTENT_1": "Yes I understand. I am Charlie, an AI housing price expert, and I will predict if the median house value for houses in a block is greater than $200,000.",
"USER_CONTENT_2": "Great! Let's begin :)",
"ASK_FOR_HYPOTHESIS": "Suppose you are trying to build a decision tree for classifying whether the median house value for households within a block is greater than $200,000. ",
"ASK_FOR_HYPOTHESIS": "Please provide a hypothesis describing the relationship between the features in the housing data and the median house value in a block being greater than $200,000. The goal is to generate a hypothesis that will be useful to a future agent performing the same prediction task. Please phrase your hypothesis as a decision tree, and phrase your hypothesis as 'Decision Tree:\n<Decision Tree>'.",
"USER_CONTENT_3": "Is the median house value for houses in this block greater than $200,000? Your answer must end with either: '(A) The median house value for houses in this block is greater than $200,000.' OR '(B) The median house value for houses in this block is less than or equal to $200,000.'",
"NO_HYPOTHESIS_USER_CONTENT_3": "Is the median house value for houses in this block greater than $200,000? You must answer either: '(A) The median house value for houses in this block is greater than $200,000.' OR '(B) The median house value for houses in this block is less than or equal to $200,000.' If you do not follow the answer template, something bad will happen.",
"LABEL_1": "(A) The median house value for houses in this block is greater than $200,000.",
"LABEL_0": "(B) The median house value for houses in this block is less than or equal to $200,000."
}

def get_train_ts_label(sample_size, train_data):
    ts_list = []
    label_list = []
    train_keys = list(train_data.keys())
    np.random.shuffle(train_keys)
    train_keys = train_keys[:sample_size]
    for key in train_keys:
        # take all keys except Label
        sample_ts = {k: v for k, v in train_data[key].items() if k != 'Label'}
        sample_label = train_data[key]['Label']
        ts_list.append(sample_ts)
        label_list.append(sample_label)
    return ts_list, label_list

def get_test_ts_label(test_data):
    return {k: v for k, v in test_data.items() if k != 'Label'}, test_data['Label']


def ts_to_string(ts_dict: dict) -> str:
    res = 'Data:\n'
    res += '\n'.join(f'{key}: {value}' for key, value in ts_dict.items())
    return res


def label_to_string(label):
    if label == 1:
        return STRINGS['LABEL_1']
    else:
        return STRINGS['LABEL_0']


def parse_response(response_string):
    # return 1 if correct, 0 if incorrect, -1 if invalid response
    if ('(A)' in response_string and '(B)' in response_string) or ('(A)' not in response_string and '(B)' not in response_string): # invalid response
        return -1
    else:
        return int('(A)' in response_string)


def create_prompt(num_shots, train_data=None, test_data=None, messages=[], train_mode=False, test_mode=False):
    # train and test not mutually exclusive
    # if only train is on, we do not include the test prompt
    # if only test is on, we do not include the train prompt (zero-shot)
    if train_mode:
        train_ts_list, train_label_list = get_train_ts_label(num_shots, train_data)
        for i in range(num_shots):
            messages.append({"role": "user", "content": ts_to_string(train_ts_list[i])})
            messages.append({"role": "assistant", "content": label_to_string(train_label_list[i])})

    if test_mode:
        test_ts, _ = get_test_ts_label(test_data)
        messages.append({"role": "user", "content": ts_to_string(test_ts)})

    return messages

def standard_prompt(train_data: dict, test_point: dict, prev_splits: list) -> str:
    #TODO: the test point still has the index and label
    res = 'Suppose you are trying to build a decision tree for classifying whether the median house value for households within a block is greater than $200,000. You have already collected the following data:\n'
    label_name = 'Median Housing Price > $200,000'
    train_data_str = ''
    for row in train_data:
        for feat, val in row.items():
            if feat != label_name:
                train_data_str += f'{feat}: {val}\n'
            else:
                label_val = 'Yes' if val == 1 else 'No'
                train_data_str += f'{label_name}: {label_val}\n'
        train_data_str += '#####\n'
    
    res += train_data_str

    res += 'You want to build a decision tree that best classifies a test point. Here is the test point:\n'
    test_point_str = ''
    for feat, val in test_point.items():
        test_point_str += f'{feat}: {val}\n'
    res += test_point_str
    res += 'To build a decision tree, you need to choose a feature to split on. '
    if len(prev_splits) > 0:
        res += 'You have already made the following splits:\n'
        for split in prev_splits:
            res += f'{split}\n'
    else:
        res += 'You have not made any splits yet.'
    
    res += 'Please suggest a new feature to split on. Please phrase your answer as follows: "{feature_name} {operator} {value}". For example, "Median Income > 35000". Be sure that the split you suggest contains the test point. Do not suggest a split you have already tried.'
    return res
