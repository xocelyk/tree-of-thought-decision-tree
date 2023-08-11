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


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


class Game24Task(Task):
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
    def __init__(self, file='housing.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'dtree', file)
        self.data = pd.read_csv(path).reset_index(inplace=True) # changed from list to dataframe
        self.value_cache = {}
        self.steps = 4 # decision tree depth
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        # TODO: we want this to return 1 if output matches true label and 0 otherwise
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            # print(sympy.simplify(expression))
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}
        
    def get_output_entropy(self, x: str, y: str) -> float:
        # This is our analog of test_output
        prev_splits = x.find('Previous Splits: ')[len('Previous Splits: '):]
        prev_splits = prev_splits.split(', ')
        prev_splits = [(feat, op, val) for feat, op, val in [split.split(' ') for split in prev_splits]]
        candidate_split = y.find('Output: ')[len('Output: '):]
        candidate_split = candidate_split.split(' ')
        candidate_split = (candidate_split[0], candidate_split[1], candidate_split[2])
        splits = prev_splits + [candidate_split]
        entropy = self.evaluate_entropy(splits)
        return entropy

    def evaluate_entropy(self, splits: list) -> float:
        # TODO: allow for different eval criteria (gini etc.)
        split_df = self.split_dataframe(splits)
        if len(split_df) == 0:
            return 0
        return -sum(split_df['label'] * split_df['label'].apply(lambda x: np.log(x)) + \
            (1 - split_df['label']) * split_df['label'].apply(lambda x: np.log(1 - x))) / len(split_df)
    
    def split_dataframe(self, splits: list) -> pd.DataFrame:
        # TODO: cache df so we do not have to repeat split each time
        df = pd.DataFrame(self.data)
        for feat, op, val in splits:
            if op == '>=':
                df = df[df[feat] >= val]
            elif op == '<=':
                df = df[df[feat] <= val]
            elif op == '<':
                df = df[df[feat] < val]
            elif op == '>':
                df = df[df[feat] > val]
            elif op == '==':
                df = df[df[feat] == val]
            else:
                raise NotImplementedError
        return df
    
            
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
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
            # print([value_last_step_prompt.format(input=x, answer=ans)])
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