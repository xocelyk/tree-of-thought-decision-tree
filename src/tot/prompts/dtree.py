import numpy as np
import pandas as pd

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

def data_dict_to_dataframe_code_string(df):
    data_dict = {col: df[col].tolist() for col in df.columns}

    # Convert dictionary to a formatted string
    dict_str = "{" + ", ".join([f"'{k}': {v}" for k, v in data_dict.items()]) + "}"

    # Return the full formatted string
    return f"pd.DataFrame({dict_str})"

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

def standard_prompt(train_data: list, test_point: dict, prev_splits: list) -> str:
    assert 'Label' not in test_point
    # turn test point to pandas dataframe
    test_point = pd.DataFrame(test_point, index=[0])
    prompt = ('You are in the process of building a decision tree to classify whether the median house value for households in a given block is greater than $200,000. Your job is to add one more split to the decision tree. You will be given three pieces of data to do this:'
              
    + '\n'
    + '\n'

    + '1. Training data: This will be presented as a pandas dataframe.' + '\n'
    + '2. Test point: This is the point whose Label you are trying to classify.' + '\n'
    + '3. Previous splits: You will be told what previous splits have been made on the data on the decision tree branch you are using for classification of the test point.'

    + '\n'
    + '\n'

    + 'The goal is to create a sequence of data splits which, when applied to the training data, will create a subset of data that is most useful for classification of the test point. Each data split will be phrased as an inequality and select one branch of the decision tree to keep. For example, if the suggested split is Feature 1 < 10, then the training data will be subsetted as df = df[df[Feature 1] < 10]. The dataframe split must contain the test point. For example, if the proposed split is Feature 1 < 10, it must be true that test_point[Feature 1] < 10. Remember, the goal is to create a subset of the training that helps us classify the test point by both selecting similar points and minimizing entropy.'

    + '\n'
    + '\n'

    + 'Here is the training data:\n'

    + data_dict_to_dataframe_code_string(train_data) + '\n' 
    + 'Here is the test point:' + '\n'
    + data_dict_to_dataframe_code_string(test_point) + '\n'
    + 'YYou will construct a decision tree by iteratively suggesting new splits in the training data, one at a time. Once all splits have been generated, for each split, we will apply the split to the dataframe, and reassign the dataframe to the side of the split that contains the test point. After all splits have been applied, we will be left with a subset of the training data. The test point will be classified as the majority label of this subset of the training data. The goal is to classify the test point correctly.'

    + '\n'
    + '\n'

    + 'You must suggest a new split of the training data. Your split must be either of the form {feature} > {value} or {feature} < {value}, where feature is a column in the training data and value is the number you want to split on.'

    + '\n'
    + '\n'

    + 'Your split must satisfy two criteria:\n'
    + '1. The split must be true of the test point. test_point[{feature}] {inequality sign} {value} must evaluate to True. For example, if test_point[feature_one] > 10, then "feature_one > 5 would be a VALID split, but "feature_one < 9 would be an INVALID split.\n'
    + '2. You must not suggest a split you have already tried.\n')
    if len(prev_splits) > 0:
        prompt += 'Here are the previous splits you have implemented: {}.'.format(prev_splits)
    else:
        prompt += 'You have not suggested any previous splits. In other words, this is the first layer in the decision tree, and you are suggesting the root. Your goal is to suggest a split that will best help classify the test point.'
    
    prompt += '\n'
    prompt += '\n'

    prompt += 'Decide on the split in three steps. First, choose the feature you want to split on. Do not choose an exact value from the test point, but instead base the choice off of the training data. Second, choose the value you want to split on. Third, choose the inequality sign you want to use for the split. The choice of inequality sign should be based on the test point. We want the inequality to point in the direction of the test point.'

    prompt += 'Propose an inequality to split the data. Select a feature and a value. Explain your thinking. Do not try to manually calculate information gain, but instead use your intuition. The last line of your response must be either "Answer: {feature} > {value}" OR "Answer: {feature} < {value}". If you do not follow the answer template, someone will die.'
    return prompt

