import numpy as np
import sys


from optrecsys import build_data
from optrecsys import build_func
from optrecsys import minimize
from opts import OPTS


def run(problem):
    fpath = OPTS['fpath'][problem]
    fkind = OPTS['fkind'][problem]
    b = OPTS['params_data'][problem]['b_list'][0]
    p = OPTS['params_data'][problem]['p_list'][0]
    s = OPTS['params_data'][problem]['s_list'][0]
    BQM = build_data(fpath, fkind, b, p, s)
    d = len(BQM)

    func = build_func(BQM)

    evals = OPTS['params_ttopt'][problem]['evals']
    rmax = OPTS['params_ttopt'][problem]['rmax']
    minimize(problem, func, d, evals, rmax)


if __name__ == '__main__':
    np.random.seed(42)

    problem = sys.argv[1] if len(sys.argv) > 1 else 'xing'
    assert problem in ['citeulike', 'movies', 'xing']

    run(problem)
