import numpy as np
import sys


from optrecsys import build_data
from optrecsys import build_func
from optrecsys import minimize
from opts import OPTS


def calc(problem, evals=5.E+6):
    rmax = OPTS['params_ttopt'][problem]['rmax']

    func, d = prep(problem)
    ttopt, res = minimize(problem, func, d, evals, rmax, with_log=True)
    res['d'] = d

    return res


def prep(problem, b=None, p=None, s=None, param_index=0):
    fpath = OPTS['fpath'][problem]
    fkind = OPTS['fkind'][problem]

    if b is None:
        b = OPTS['params_data'][problem]['b_list'][param_index]
    if p is None:
        p = OPTS['params_data'][problem]['p_list'][param_index]
    if s is None:
        s = OPTS['params_data'][problem]['s_list'][param_index]

    BQM = build_data(fpath, fkind, b, p, s)
    d = len(BQM)

    func = build_func(BQM)

    return func, d


if __name__ == '__main__':
    np.random.seed(42)

    res = {}
    for problem in ['xing', 'movies', 'citeulike']:
        print(f'\n\n--- Solve {problem}')
        res[problem] = calc(problem)
        np.savez_compressed('./result/result.npz', res=res)

    print(f'\n\n')
