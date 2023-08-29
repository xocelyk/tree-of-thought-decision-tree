import itertools
import numpy as np
from functools import partial
from tot.models import gpt, chatgpt

GET_SPLIT_FUNC = {
        'name': 'extract_data_split',
        'description': 'Takes a feature, operator, an value, and returns a string representing the inequality',
        'parameters': {
            'type': 'object',
            'properties': {
                'feature': {
                    'type': 'string',
                    'description': 'Feature in the dataframe'
                },
                'operator': {
                    'type': 'string',
                    'description': 'Inequality operator: <, >, <=, or >='
                },
                'value': {
                    'type': 'number',
                    'description': 'Value to compare against'
                }
            },
            'required': ['feature', 'operator', 'value'],
        }
    }
    

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    # check if task name is dtree
    if not task.name == 'dtree': # TODO: fix the if else logic and add name to each task
        value_prompt = task.value_prompt_wrap(x, y)
        if cache_value and value_prompt in task.value_cache:
            return task.value_cache[value_prompt]
        value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
        value = task.value_outputs_unwrap(x, y, value_outputs)
        if cache_value:
            task.value_cache[value_prompt] = value
        return value
    elif task.name == 'dtree':
        # n_evaluate_sample does not matter in this case
        value = task.get_output_info_gain(x, y)
        return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if ', '.join(y) in local_value_cache:  # avoid duplicate candidates
            value = -float('inf')
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[', '.join(y)] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, test_point: dict, prev_splits, n_generate_sample, prompt_sample, stop, func=None) -> list:
    # x is test point
    # y is previous splits
    # if prompt_sample == 'standard':
    #     prompt = task.standard_prompt_wrap(test_point, prev_splits)
    # elif prompt_sample == 'cot':
    #     prompt = task.cot_prompt_wrap(test_point, prev_splits)
    # else:
    #     raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = []
    for i in range(n_generate_sample):
        feature_prompt = task.ask_for_feature_prompt_wrap(prev_splits)
        feature = task.parse_feature_response(gpt(feature_prompt, n=1, stop=stop)[0])
        value_prompt = task.ask_for_value_prompt_wrap(feature, test_point, prev_splits)
        value = gpt(value_prompt, n=1, stop=stop)[0]
        print('value: ', value)
        value = task.parse_value_response(value)
        inequality_sign = task.get_inequality_sign(test_point, feature, value)
        candidate = feature + ' ' + inequality_sign + ' ' + str(value)
        print('candidate: ', candidate)
        samples.append(candidate)
    print('samples: ', samples)
    return [prev_splits + [s] for s in samples]

def solve(args, task, idx, to_print=True):
    # TODO: if leaf is pure, freeze it
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    print('input: ', x)
    print()
    ys = [[]]  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    input_label = task.get_input_label(idx)
    pred = task.classify(ys)
    print(f'input: {input_label}\npred: {pred}\n')
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}