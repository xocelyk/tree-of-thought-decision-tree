import re
import os
import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.dtree import * 
import numpy as np

'''
We don't use value prompt wrap or value outputs unwrap, instead taking the output and using
it to split dataframe and calculate entropy. Entropy is the output.
'''


class DTreeTask(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='housing.csv', num_train=40):
        """
        file: a csv file (fixed)
        """
        self.name = 'dtree'
        super().__init__()
        path = os.path.join(DATA_PATH, 'dtree', file)
        self.data = pd.read_csv(path) # changed from list to dataframe
        self.data.dropna(inplace=True) # TODO: this should probably happen before the class is initialized
        random_state = 42
        self.train_data = self.data.sample(num_train, random_state=random_state)
        self.train_data.index = range(num_train)
        self.data = self.data.drop(self.train_data.index).reset_index()
        self.data.index = range(len(self.data))
        self.value_cache = {}
        self.steps = 3 # decision tree depth
        self.stops = ['\n'] * 3 # TODO: what does this do?

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        series = self.data.iloc[idx]
        # to dict, don't include idx or label
        # TODO: might need to recover label in the future for evaluation
        series = series.drop(['index', 'Label']).to_dict()
        return series

    def test_output(self, idx: int, output: str):
        candidate_splits = self.parse_splits(output)
        split_df = self.data[idx].split(candidate_splits)
        # majority vote over dtree branch
        majority_label = np.argmax(np.bincount(split_df['label']))
        # return 1 if majority label is correct, 0 otherwise
        return {'r': int(majority_label == self.data[idx]['label'])}
        
    def get_output_info_gain(self, x: str, y: str) -> float:
        # TODO: need to add some stopping criterion for when we are already at minimum entropy
        # This is our analog of test_output
        splits = self.parse_splits(y)
        entropy = self.evaluate_info_gain(splits)
        return entropy 

    def parse_splits(self, y: list) -> list:
        res = []
        operator_signs = '<= >= < > =='.split(' ')
        # TODO: fix the splitting logic
        for candidate_split in y:
            # separate into feature, operator, value
            for op in operator_signs:
                if op in candidate_split:
                    feat, val = [el.strip() for el in candidate_split.split(op)]
                    break
            try:
                val = float(val)
            except:
                # TODO: this is happening too much
                print(candidate_split, feat, op, val)
                raise ValueError(f'Could not parse {candidate_split}')
            res.append((feat, op, val))
        return res

    def evaluate_info_gain(self, splits: list) -> float:
        # TODO: this calculation needs to change and account for the total information added on both sides
        # it is not optimal to just split so that we get one side of the data with a single data point and minimum entropy, this is silly!
        df = self.train_data.copy()
        if len(splits) == 1:
            prev_df = df
        else:
            prev_splits = splits[:-1]
            prev_df = self.split_dataframe(df, prev_splits)

        cur_split = [splits[-1]]
        cur_df_1, cur_df_2 = self.split_dataframe(prev_df, cur_split, two_sided=True)
        if cur_df_1.empty or cur_df_2.empty:
            return 0
        cur_entropy_1 = self.calculate_entropy(cur_df_1)
        cur_entropy_2 = self.calculate_entropy(cur_df_2)
        cur_entropy = (cur_entropy_1 * len(cur_df_1) + cur_entropy_2 * len(cur_df_2)) / (len(cur_df_1) + len(cur_df_2))
        prev_entropy = self.calculate_entropy(prev_df)
        info_gain = prev_entropy - cur_entropy
        print(splits, prev_entropy, cur_entropy, info_gain)
        print('T/F Split on prev df: {}/{}'.format(len(prev_df[prev_df['Label'] == 1]), len(prev_df[prev_df['Label'] == 0])))
        print('T/F Split on cur df 1: {}/{}'.format(len(cur_df_1[cur_df_1['Label'] == 1]), len(cur_df_1[cur_df_1['Label'] == 0])))
        print('T/F Split on cur df 2: {}/{}'.format(len(cur_df_2[cur_df_2['Label'] == 1]), len(cur_df_2[cur_df_2['Label'] == 0])))
        print()
        return info_gain


    @staticmethod
    def calculate_entropy(df: pd.DataFrame) -> float:
        p_plus = df['Label'].mean()
        p_minus = 1 - p_plus

        # Handle cases where p_plus or p_minus are 0, as log2(0) is undefined
        if p_plus == 0 or p_minus == 0:
            return 0

        # Calculate entropy
        entropy = -p_plus * np.log2(p_plus) - p_minus * np.log2(p_minus)

        return entropy
    
    @staticmethod
    def split_dataframe(df, splits: list, two_sided = False) -> pd.DataFrame:
        for feat, op, val in splits:
            if op == '>=':
                res = df[df[feat] >= val]
            elif op == '<=':
                res = df[df[feat] <= val]
            elif op == '<':
                res = df[df[feat] < val]
            elif op == '>':
                res = df[df[feat] > val]
            elif op == '==':
                res = df[df[feat] == val]
            else:
                raise NotImplementedError
        if two_sided:
            return res, df[~df.index.isin(res.index)]
        else:
            return res
    
    def standard_prompt_wrap(self, x: dict, prev_splits: str='') -> str:
        train_data = self.train_data.copy().to_dict(orient='records')
        prompt = standard_prompt(train_data, x, prev_splits) # TODO: where do prev_splits come from?
        return prompt

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
        else:
            prompt = propose_prompt.format(input=current_numbers)
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        # TODO: this will not use a prompt but actually invoke the oracle
        # assume self.data has a label called "label"
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value