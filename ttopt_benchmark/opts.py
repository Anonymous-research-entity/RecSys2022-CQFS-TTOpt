# Replace this code with the correct path to data:
with open('./path.txt', 'r') as f:
    FPATH_PREFIX = f.readline().replace('\n', '')


OPTS = {
    'fpath': {
        'citeulike': FPATH_PREFIX + '/CiteULike_a_FPM',
        'movies': FPATH_PREFIX + '/TheMoviesDataset_FPM',
        'xing': FPATH_PREFIX + '/XingChallenge2017_FPM',
    },
    'fkind': {
        'citeulike': 'npz',
        'movies': 'npz',
        'xing': 'npy',

    },
    'params_data': {
        'citeulike': {
            'b_list': [1e-4, 1e-4, 1e-4, 1e-4, 1e-2, 1e-3],
            'p_list': [20, 30, 40, 60, 80, 95],
            's_list': [1e2, 1e2, 1e2, 1e2, 1e2, 1e3],
        },
        'movies': {
            'b_list': [1e-4, 1e-4, 1e-3, 1e-4, 1e-3, 1e-3],
            'p_list': [20, 30, 40, 60, 80, 95],
            's_list': [1e2, 1e3, 1e2, 1e2, 1e2, 1e3],
        },
        'xing': {
            'b_list': [1e-3, 1e-4, 1e-3, 1e-4],
            'p_list': [40, 60, 80, 95],
            's_list': [1e3, 1e1, 1e2, 1e3],
        },
    },
    'params_ttopt': {
        'citeulike': {
            'evals': 3.E+5,
            'rmax': 4,
        },
        'movies': {
            'evals': 1.E+5,
            'rmax': 4,
        },
        'xing': {
            'evals': 5.E+3,
            'rmax': 2,
        },
    }
}
