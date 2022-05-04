import re
import time

import numpy as np

from recsys.Base.DataIO import DataIO
from utils.naming import get_experiment_id
from utils.statistics import similarity_statistics, BQM_statistics


class CQFSTT:
    SAVED_MODELS_FILE = 'saved_CQFSTT_models.zip'
    STATISTICS_FILE = 'statistics'
    TIMINGS_FILE = 'timings'

    def __init__(self, ICM_train, S_CF, S_CBF, base_folder_path, statistics=None, *, sampler):

        self.n_items, self.n_features = ICM_train.shape
        self.ICM_train = ICM_train.copy()

        if re.match('.*/.*ICM.*/.*Recommender.*/', base_folder_path) is None:
            self.__print("[WARNING] base_folder_path has a custom format, we suggest to use the following one for "
                         "compatibility with other classes:\n"
                         "DatasetName/ICMName/CFRecommenderName/")

        self.base_folder_path = base_folder_path if base_folder_path[-1] == '/' else f"{base_folder_path}/"
        self.dataIO = DataIO(self.base_folder_path)

        self.statistics = {}
        if statistics is not None:
            self.statistics = statistics
        self.timings = self.__load_timings()

        ##################################################
        # Model variables

        # self.IPMs = {}
        self.FPMs = {}
        # self.BQMs = {}
        self.selections = {}

        ##################################################
        # Load previously saved models

        self.saved_models = self.__load_previously_saved_models()

        if not self.saved_models['FPM_K'] or not self.saved_models['FPM_E']:
            ##################################################
            # Model initialization

            self.__print("Building the base models...")
            base_model_time = time.time()

            # self.S_CF = S_CF.copy()
            # self.S_CBF = S_CBF.copy()
            assert S_CF.shape == S_CBF.shape, "The two sparse matrices have different shapes."
            assert S_CF.shape == (self.n_items, self.n_items), "The similarity matrices do not have the right shape."

            S_CF.data = np.ones_like(S_CF.data)
            S_CBF.data = np.ones_like(S_CBF.data)

            S_CF_bool = S_CF.astype(np.bool)
            S_CBF_bool = S_CBF.astype(np.bool, copy=True)

            #########################
            # Compute the bonus for similarities in common

            K_time = time.time()
            K = S_CBF.multiply(S_CF_bool)
            K.data = -K.data
            K_time = time.time() - K_time

            #########################
            # Compute the penalization for similarities not in common

            E_time = time.time()
            S_intersection = S_CBF_bool.multiply(S_CF_bool)
            E = S_CBF_bool - S_intersection
            E = S_CBF.multiply(E)
            E_time = time.time() - E_time

            assert K.nnz + E.nnz == S_CBF.nnz, "The number of items to keep and to penalize is not correct."

            S_union = S_CF_bool + S_CBF_bool

            self.statistics = similarity_statistics(S_CF, S_CBF, S_intersection, S_union, statistics=self.statistics)
            # self.statistics = error_statistics(S_CF, S_CBF, N=S_union.nnz, suffix="_dot", statistics=self.statistics)
            self.__save_statistics()

            base_model_time = time.time() - base_model_time
            self.timings['base_model_time'] = base_model_time
            self.timings['K_time'] = K_time
            self.timings['E_time'] = E_time
            self.timings['avg_response_time'] = {}
            self.timings['n_select_experiments'] = {}

            self.FPM_K, IPM_K_time, FPM_K_time, QUBO_K_time = self._compute_fpm(K)
            self.FPM_E, IPM_E_time, FPM_E_time, QUBO_E_time = self._compute_fpm(E)
            IPM_time = IPM_K_time + IPM_E_time
            FPM_time = FPM_K_time + FPM_E_time
            QUBO_time = QUBO_K_time + QUBO_E_time

            self.timings['avg_IPM_time'] = IPM_time
            self.timings['avg_FPM_time'] = FPM_time
            self.timings['avg_QUBO_time'] = QUBO_time
            self.timings['n_fit_experiments'] = 1

            self.__save_timings()

            self.saved_models['FPM_K'] = True
            self.saved_models['FPM_E'] = True

            self.dataIO.save_data(self.SAVED_MODELS_FILE, self.saved_models)
            self.dataIO.save_data('K', {'K': K})
            self.dataIO.save_data('E', {'E': E})
            self.dataIO.save_data('FPM_K', {'FPM_K': self.FPM_K})
            self.dataIO.save_data('FPM_E', {'FPM_E': self.FPM_E})

            self.__print("Base models successfully built.")
        else:
            self.__print("Base models successfully loaded.")

        ##################################################
        # Solver initialization

        self.solver = sampler
        self.selection_type = 'cqfs_tt'

        if self.timings['avg_response_time'].get(self.selection_type) is None:
            self.timings['avg_response_time'][self.selection_type] = 0
            self.timings['n_select_experiments'][self.selection_type] = 0

    @staticmethod
    def __print(msg):
        print(f"CQFSTT: {msg}")

    def __save_statistics(self):
        self.dataIO.save_data(self.STATISTICS_FILE, self.statistics)

    def __save_timings(self):
        self.dataIO.save_data(self.TIMINGS_FILE, self.timings)

    def __load_timings(self):
        timings = {}
        try:
            timings = self.dataIO.load_data(self.TIMINGS_FILE)
        except FileNotFoundError:
            self.__print("No timings file found.")
        return timings

    def __load_base_model(self, model):

        model_file = f'{model}.zip'

        try:
            if model == 'FPM_K':
                self.FPM_K = self.dataIO.load_data(model_file)['FPM_K']
            elif model == 'FPM_E':
                self.FPM_E = self.dataIO.load_data(model_file)['FPM_E']
            return True

        except FileNotFoundError:
            return False

    def __load_previously_saved_models(self):

        self.__print("Trying to load previously saved models.")

        saved_models = {
            'FPM_K': False,
            'FPM_E': False,
        }

        try:
            saved_models = self.dataIO.load_data(self.SAVED_MODELS_FILE)

            for model in saved_models:
                if saved_models[model]:
                    saved_models[model] = self.__load_base_model(model)

        except FileNotFoundError:
            self.__print("No model saved for this set of experiments.")

        self.dataIO.save_data(self.SAVED_MODELS_FILE, saved_models)
        return saved_models

    @staticmethod
    def __p_to_k(p, n_features):
        assert p is not None, "Please, choose a selection percentage." \
                              "The value should be between 0 and 1 or between 0 and 100."

        if 1 < p <= 100:
            p /= 100
        elif p > 100 or p < 0:
            raise ValueError("Percentage value should be between 0 and 1 or between 0 and 100.")

        return n_features * p

    def _compute_fpm(self, IPM):
        QUBO_time = time.time()
        IPM_time = 0

        ICM = self.ICM_train.astype(np.float64)
        IPM = IPM.astype(np.float64)

        FPM_time = time.time()
        IFPM = IPM * ICM
        IFPM.eliminate_zeros()
        FPM = ICM.T * IFPM
        FPM.eliminate_zeros()
        FPM_time = time.time() - FPM_time
        QUBO_time = time.time() - QUBO_time

        return FPM, IPM_time, FPM_time, QUBO_time


    def fit(self, alpha=1, beta=1, vartype='BINARY', save_FPM=False):

        fitID = get_experiment_id(alpha, beta)

        self.__print(f"[{fitID}] Fitting experiment.")

        if self.FPMs.get(fitID) is None:
        # if self.BQMs.get(fitID) is None:
            # IPM = self.IPMs.get(fitID)
            # if IPM is None:
            #     IPM = alpha * self.K + beta * self.E

            FPM = alpha * self.FPM_K + beta * self.FPM_E

            experiment_dataIO = self.__get_experiment_dataIO(fitID)
            FPM_statistics_file = 'FPM_statistics'
            try:
                experiment_dataIO.load_data(FPM_statistics_file)
                self.__print(f"[{fitID}] Found a previously saved {FPM_statistics_file} file."
                             f" Skipping statistics computation.")
            except FileNotFoundError:

                linear = FPM.diagonal()

                quadratic = FPM.tocsr()
                quadratic.setdiag(0)
                quadratic.eliminate_zeros()
                quadratic_data = quadratic.data

                FPM_statistics = BQM_statistics(linear, quadratic_data, FPM.shape, prefix='FPM_')
                self.__print(f"[{fitID}] Computed {FPM_statistics_file}. Saving to file.")
                experiment_dataIO.save_data(FPM_statistics_file, FPM_statistics)

            FPM = np.array(FPM.todense())

            if save_FPM:
                FPM_file = 'FPM'
                try:
                    experiment_dataIO.load_data(FPM_file)
                    self.__print(f"[{fitID}] Found a previously saved {FPM_file} file."
                                 f" Skipping FPM saving.")
                except FileNotFoundError:
                    self.__print(f"[{fitID}] Saving FPM to file...")
                    experiment_dataIO.save_data(FPM_file, {'FPM': FPM})

            # self.IPMs[fitID] = IPM.copy()
            self.__print(f"[{fitID}] FPM shape {FPM.shape}")
            self.FPMs[fitID] = FPM
            # self.BQMs[fitID] = BQM.copy()

        # else:
        #     FPM = self.FPMs[fitID]
        #     BQM = dimod.as_bqm(FPM.toarray(), vartype)
        #
        #     self.BQMs[fitID] = BQM.copy()

    def fit_many(self, alphas, betas, vartype='BINARY', save_FPM=False):

        for alpha in alphas:
            for beta in betas:
                self.fit(alpha, beta, vartype, save_FPM)

    def __get_selection_from_sample(self, sample):

        selection = np.zeros(self.n_features, dtype=bool)
        for k, v in sample.items():
            if v == 1:
                ind = int(k)
                selection[ind] = True

        return selection

    def __get_experiment_dataIO(self, expID):
        experiment_folder_path = f"{self.base_folder_path}{expID}/"
        return DataIO(experiment_folder_path)

    def __get_selection_dataIO(self, expID):
        selection_folder_path = f"{self.base_folder_path}{expID}/{self.selection_type}/"
        return DataIO(selection_folder_path)

    def __save_selection(self, expID, selection):

        selection_dataIO = self.__get_selection_dataIO(expID)
        selection_dict = {
            'selection': selection,
        }
        selection_dataIO.save_data(self.selection_type, selection_dict)
        self.__print(f"[{expID}] Selection saved.")

    def __load_selection(self, expID):

        selection_dataIO = self.__get_selection_dataIO(expID)

        try:
            selection = selection_dataIO.load_data(self.selection_type)['selection']
            self.__print(f"[{expID}] Found an existing selection.")
            self.selections[expID] = selection.copy()

        except FileNotFoundError:
            self.__print(f"[{expID}] No previous selection found.")

    def _bqm_from_fpm(self, fpm: np.ndarray, *, s: float, k: float):
        assert fpm.shape[0] == fpm.shape[1]
        F = len(fpm)
        fpm_diag = np.diag(np.diag(fpm))
        fpm_other = fpm - fpm_diag
        bqm = fpm_diag + np.triu(fpm_other + fpm_other.T)
        bqm = (
            bqm
            - 2 * k * s * np.eye(F)
            + s * (2 * np.triu(np.ones((F, F))) - np.eye(F))
        )
        inv_scalar = max(-bqm.min(), bqm.max())
        bqm /= inv_scalar
        return bqm

    def select(self, alpha, beta, combination_strength=1):
        raise NotImplementedError("Method not implemented yet.")

    def select_p(self, p, alpha, beta, combination_strength=1, vartype='BINARY', save_FPM=False):
        k = self.__p_to_k(p, self.n_features)

        expID = get_experiment_id(alpha, beta, p=p, combination_strength=combination_strength)
        self.__load_selection(expID)
        if self.selections.get(expID) is not None:
            return self.selections[expID].copy()

        self.__print(f"[{expID}] Starting selection.")

        FPM = self.FPMs.get(expID)
        if FPM is None:
            fitID = get_experiment_id(alpha, beta)
            FPM = self.FPMs.get(fitID)
            if FPM is None:
                self.fit(alpha=alpha, beta=beta, vartype=vartype, save_FPM=save_FPM)
                FPM = self.FPMs.get(fitID)

        BQM = self._bqm_from_fpm(FPM, k=k, s=combination_strength)

        self.__print(f"[{expID}] Sampling the problem.")
        response_time = time.time()
        best_sample = self.solver.sample(BQM)
        response_time = time.time() - response_time

        experiment_timings = {
            'response_time': response_time,
        }
        selection_dataIO = self.__get_selection_dataIO(expID)
        selection_dataIO.save_data(self.TIMINGS_FILE, experiment_timings)

        n_experiments = self.timings['n_select_experiments'][self.selection_type]
        total_response_time = self.timings['avg_response_time'][self.selection_type] * n_experiments
        n_experiments += 1
        self.timings['n_select_experiments'][self.selection_type] = n_experiments
        self.timings['avg_response_time'][self.selection_type] = (total_response_time + response_time) / n_experiments
        self.__save_timings()

        selection = self.__get_selection_from_sample(best_sample)

        self.__print(f"[{expID}] Selected {selection.sum()} features in {response_time} sec.")

        self.selections[expID] = selection.copy()
        self.__save_selection(expID, selection)
        return selection

    def select_many_p(self, ps, alphas, betas, combination_strengths, vartype='BINARY', save_FPMs=False, parameter_product=True):
        if parameter_product:
            for p in ps:
                for alpha in alphas:
                    for beta in betas:
                        for combination_strength in combination_strengths:
                            self.select_p(p, alpha, beta, combination_strength, vartype, save_FPMs)
        else:
            args_zip = zip(ps, alphas, betas, combination_strengths)
            for args in args_zip:
                self.select_p(args[0], args[1], args[2], args[3], vartype, save_FPMs)
