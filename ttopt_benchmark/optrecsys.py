import numpy as np
from scipy.sparse import load_npz
from time import perf_counter as tpc
from ttopt import TTOpt


def build_BQM(FPM, p, s):
    assert FPM.shape[0] == FPM.shape[1]

    F = len(FPM)
    k = p / 100 * F

    FPM_diag = np.diag(np.diag(FPM))
    FPM_other = FPM - FPM_diag

    BQM = FPM_diag + np.triu(FPM_other + FPM_other.T)
    BQM = (
        BQM
        - 2 * k * s * np.eye(F)
        + s * (2 * np.triu(np.ones((F, F))) - np.eye(F))
    )

    inv_scalar = max(-BQM.min(), BQM.max())
    BQM /= inv_scalar

    return BQM


def build_FPM(FPM_K, FPM_E, b):
    return FPM_K + b * FPM_E


def build_data(fpath, fkind, b, p, s):
    FPM_K = load_numpy_file(fpath + '/FPM_K', fkind)
    FPM_E = load_numpy_file(fpath + '/FPM_E', fkind)

    FPM = build_FPM(FPM_K, FPM_E, b)
    BQM = build_BQM(FPM, p, s)

    return BQM


def build_func(BQM):
    def func(X):
        return ((X @ BQM) * X).sum(axis=1)

    return func


def load_numpy_file(fpath, fkind='npz'):
    if fkind == 'npz':
        return load_npz(fpath + f'.{fkind}').todense()
    elif fkind == 'npy':
        return np.load(fpath + f'.{fkind}', allow_pickle=False)


def minimize(problem, func, d, evals, rmax, text_add='', with_log=True):
    res = {'k': [], 'm': [], 't': [], 'y': []}
    t0 = tpc()

    def callback(info):
        """Function is called whenever the optimization result is improved."""
        y_min = info['last'][1] # Current found optimum
        i_min = info['last'][2] # Multi-index, which relates to current optimum
        evals = info['last'][4] # Current number of the target function calls
        t = tpc() - t0
        k = np.sum(i_min)

        res['k'].append(k)
        res['m'].append(evals)
        res['t'].append(t)
        res['y'].append(y_min)

        if not with_log:
            return

        text = 'LOG |'
        text += f' evals: {evals:-7.1e} |'
        text += f' selected indices: {k:-5d} |'
        text += f' y_min: {y_min:-14.8f} |'
        text += f' time: {t:-7.3f}'
        print(text)

    ttopt = TTOpt(f=func, d=d, n=2, name=problem, evals=evals, is_func=False,
        callback=callback, with_log=False)
    ttopt.minimize(rmax=rmax)

    if with_log:
        text = '-'*70 + '\n' + ttopt.info()
    else:
        i_min = ttopt.i_min
        text = ttopt.info()
        text += f'| i-num: {np.sum(i_min):-4d}'
    print(text + text_add)

    return ttopt, res
