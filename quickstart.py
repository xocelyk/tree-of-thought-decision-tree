import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
from tot.tasks.dtree import DTreeTask

# args = argparse.Namespace(backend='gpt-3.5-turbo', temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
args = argparse.Namespace(backend='gpt-3.5-turbo', temperature=0.7, task='dtree', naive_run=False, prompt_sample='standard', method_generate='sample', method_evaluate='value', method_select='greedy', n_generate_sample=4, n_evaluate_sample=4, n_select_sample=3)


task = DTreeTask()
for i in range(200, 205):
    ys, infos = solve(args, task, i)
    print(ys)