from functools import partial
from typing import Union

import numpy as np
from ttopt import TTOpt


class CQFSTTSampler:
    _rmax: int
    _evals: int
    _verbose: bool

    def __init__(
            self,
            rmax: int = 10,
            evals: Union[float, int] = 1e6,
            verbose: bool = False,
    ):
        self._rmax = rmax
        self._evals = int(evals)
        self._verbose = verbose

    def _func(self, x, *, bqm):
        return ((x @ bqm) * x).sum(axis=1)

    def sample(self, bqm: np.ndarray):
        F = len(bqm)
        ttopt = TTOpt(
            f=partial(self._func, bqm=bqm),
            d=F,
            n=2,
            evals=self._evals,
            name='Tensor',
            is_func=False,
            with_log=self._verbose,
        )
        ttopt.minimize(self._rmax)
        return dict(enumerate(ttopt.x_min))
