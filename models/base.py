import numpy as np
import itertools
from functools import wraps
import negmas

NUMPY_TYPE = 'float16'

def cartesian_product_pp(arrays, out=None):
    la = len(arrays)
    L = *map(len, arrays), la
    dtype = np.result_type(*arrays)
    arr = np.empty(L, dtype=dtype)
    arrs = *itertools.accumulate(itertools.chain((arr,), itertools.repeat(0, la-1)), np.ndarray.__getitem__),
    idx = slice(None), *itertools.repeat(None, la-1)
    for i in range(la-1, 0, -1):
        arrs[i][..., i] = arrays[i][idx[:la-i]]
        arrs[i-1][1:] = arrs[i]
    arr[..., 0] = arrays[0][idx]
    return arr.reshape(-1, la)

def generate_uniform_hypotheses(n, k):
    
    def recursive_combinations(level, remaining, current_combination, results):
        if level == n - 1:
            current_combination.append(remaining)
            results.append(current_combination)
            return
        for i in range(remaining + 1):
            new_combination = current_combination.copy()
            new_combination.append(i)
            recursive_combinations(level + 1, remaining - i, new_combination, results)

    results = []
    recursive_combinations(0, k, [], results)
    return np.array(results) / k

def likelihood_none_zero(likelihood_cal):
    @wraps(likelihood_cal)
    def wraplikelihood(*args, **kw):
        likelihood = likelihood_cal(*args, **kw)
        if likelihood.max() == 0:
            likelihood[:] = 1
            print('we are in the wrap')
        likelihood = (likelihood.shape[0] * likelihood) / likelihood.sum()
        return likelihood.astype(NUMPY_TYPE)
    return wraplikelihood

class weight_space:

    def __init__(self, ufun, normal_part = False, mode = '1g', num_hypothesis = 11) -> None:

        self.num_issues = len(ufun.weights)

        self.num_hypothesis = num_hypothesis
        self.h_space = None
        self.h_probs = None
        self.expectation = None
        self.temp_h_probs = None
        self.flatten_accume = None
        self.accume = None
        self.normal_part = normal_part
        self.mode = mode

        self.issue_values_range = {}
        self.issue_num_values = {}
        _i = 0
        _ii = 0
        for k in range(len(ufun.issues)):
            self.issue_values_range[k] = [_ii]
            _i = len(ufun.issues[k].values)
            self.issue_num_values[k] = _i
            _ii = _ii + _i
            self.issue_values_range[k].append(_ii)
        self.num_values = _ii

        self._initial()
        self.update_accume()

    def _initial(self):
        if self.mode[0] == '0':
            self.get_expectation = self._get_expectation_0
            self.update = self._update_0
            if self.mode[1] == 'g':
                Hs, Ps = self._Grid_0_WHS()
            elif self.mode[1] == 'r':
                Hs, Ps = self._Rank_0_WHS()
        elif self.mode[0] == '1':
            self.get_expectation = self._get_expectation_1
            self.update = self._update_1
            if self.mode[1] == 'g':
                Hs, Ps = self._Grid_1_WHS()
            elif self.mode[1] == 'r':
                Hs, Ps = self._Rank_1_WHS()
            if self.normal_part == True:
                self.expectation_except_issue_i = self._expectation_except_issue_i_normalize
        self.h_space = Hs
        self.h_probs = Ps.copy()
        self.initial_h_probs = Ps.copy()

    def flatten_weights_HS(self, W_Hspace):
        len_E_HS = self.num_values
        weights_flatten = np.ones([W_Hspace.shape[0], len_E_HS], dtype = NUMPY_TYPE)
        for i in range(self.num_issues):
            range_se = self.issue_values_range[i]
            range_s = range_se[0]
            range_e = range_se[1]
            repeat_times = self.issue_num_values[i]
            weights_flatten[:, range_s:range_e] = np.repeat(W_Hspace[:,i].reshape([-1,1]), repeat_times, axis = 1)
        return weights_flatten
      
    def _get_expectation_1(self):
        accume_w = (self.h_space * self.h_probs).sum(axis = 1) / self.h_probs.sum(axis = 1)
        sum_w = accume_w.sum()
        accume_w = accume_w / sum_w
        return accume_w
    
    def update_accume(self):
        self.accume = self.get_expectation()
        self.flatten_accume = self.flatten_weights_HS(self.accume.reshape([1,-1])).reshape([-1])

    def expectation_except_issue_i(self, issue_i):
        wh_issue_i = self.h_space[issue_i, :]
        # prob_issue_i = self.h_probs[issue_i, :]
        expectation = np.copy(self.accume)
        expectation = expectation.reshape([1,-1]).repeat(repeats = wh_issue_i.shape[0], axis = 0)
        expectation[:, issue_i] = wh_issue_i
        # normalize before evluation
        # expectation = expectation / expectation.sum(axis = 1).reshape([-1,1])
        return expectation
    
    def _expectation_except_issue_i_normalize(self, issue_i):
        wh_issue_i = self.h_space[issue_i, :]
        # prob_issue_i = self.h_probs[issue_i, :]
        expectation = np.copy(self.accume)
        expectation = expectation.reshape([1,-1]).repeat(repeats = wh_issue_i.shape[0], axis = 0)
        expectation[:, issue_i] = wh_issue_i
        # normalize before evluation
        expectation = expectation / expectation.sum(axis = 1).reshape([-1,1])
        return expectation
    
    def _Grid_1_WHS(self):
        weights_HS = np.linspace(0, 1, self.num_hypothesis).reshape([1,-1]).repeat(axis = 0, repeats = self.num_issues)
        WH_probs = np.ones_like(weights_HS)
        return weights_HS, WH_probs

    def _update_1(self, accume_e, likelihood_func):
        for i in range(self.num_issues):
            hs_i = self.expectation_except_issue_i(i)
            hs_i_flatten = self.flatten_weights_HS(hs_i)
            ufun_flatten = hs_i_flatten * accume_e
            Ls = likelihood_func(ufun_flatten)
            h_probs_i = self.h_probs[i, :] * Ls

            prob_sum = h_probs_i.sum()
            if prob_sum != 0:
                self.h_probs[i, :] = (h_probs_i * h_probs_i.size) / prob_sum
            else:
                self.h_probs[i, :] = np.ones_like(h_probs_i)
            self.update_accume()

    def update_recursive(self, accume_e, likelihood_func):
        for i in range(self.num_issues):
            hs_i = self.expectation_except_issue_i(i)
            hs_i_flatten = self.flatten_weights_HS(hs_i)
            ufun_flatten = hs_i_flatten * accume_e
            Ls = likelihood_func(ufun_flatten)
            h_probs_i = self.h_probs[i, :] * Ls

            prob_sum = h_probs_i.sum()
            if prob_sum != 0:
                self.h_probs[i, :] = (h_probs_i * h_probs_i.size) / prob_sum
            else:
                self.h_probs[i, :] = np.ones_like(h_probs_i)
        self.update_accume()

    def refresh_probs(self, probs):
        self.h_probs = probs
        self.update_accume()
    
    def resample_inverse(self, gate):
        num_p = self.num_hypothesis
        for j in range(self.num_issues):
            h_j = self.h_space[j, :]
            p_j = self.h_probs[j, :]
            p_j = p_j / p_j.sum()
            N_eff = 1 / np.sum(p_j**2)
            if N_eff >= num_p * gate:
                continue
            # Compute the CDF
            cdf = np.cumsum(p_j)
            uniform_samples = np.linspace(0.05, 1, num_p)

            def inverse_sample(u, cdf, h):
                for i in range(len(cdf)):
                    if u <= cdf[i]:
                        y1 = h[i]
                        if i == len(cdf) - 1:  # If it's the last element
                            y2 = h[i]
                        else:
                            y2 = h[i+1]
                        if i == 0:
                            x1 = 0
                        else:
                            x1 = cdf[i-1]
                        x2 = cdf[i]
                        return y1 + (y2 - y1) * (u - x1) / (x2 - x1)
                return h[-1]
            # Generate samples using inverse sampling
            generated_samples = [inverse_sample(u, cdf, h_j) for u in uniform_samples]
            # print('gs:', generated_samples)
            final_samples = []
            i = 0
            processed = set()  # To keep track of already processed samples
            while i < len(generated_samples):
                sample = generated_samples[i]
                if sample not in processed:
                    count = generated_samples.count(sample)
                    if count == 1:
                        final_samples.append(sample)
                        i += 1
                    else:
                        idx = h_j.tolist().index(sample)
                        start = h_j[idx - 1] if idx - 1 >= 0 else h_j[idx]
                        end = h_j[idx + 1] if idx + 1 < len(h_j) else h_j[idx]
                        step = (end - start) / (count + 1)
                        
                        # Add one occurrence of the original sample
                        final_samples.append(sample)
                        
                        # Interpolate for the remaining samples
                        for _j in range(1, count):
                            final_samples.append(start + step * (_j + 1))
                        
                        i += count  # Skip past all occurrences of the current sample
                        processed.add(sample)  # Mark the sample as processed
                else:
                    i += 1
            # print('fs', final_samples)
            final_samples = np.sort(final_samples)
            

            self.h_space[j, :] = final_samples
            self.h_probs[j, :] = np.ones(self.num_hypothesis)

class evaluation_space:

    def __init__(self, ufun, mode = '2g', num_hypothesis = 11, normal = False) -> None:
        self.h_space = None
        self.h_probs = None
        self.expectation = None
        self.num_values = None
        self.issue_num_values = {}
        self.issue_values_range = {}
        self.num_issues = len(ufun.issues)
        self.accume = None
        self.mode = mode
        self.normal = normal
        _i = 0
        _ii = 0
        for k in range(len(ufun.issues)):
            self.issue_values_range[k] = [_ii]
            _i = len(ufun.issues[k].values)
            self.issue_num_values[k] = _i
            _ii = _ii + _i
            self.issue_values_range[k].append(_ii)
        self.num_values = _ii
        
        self.num_hypothesis = num_hypothesis

        self.initial()
        self.update_accume()

    def initial(self):
        if self.mode == '2g':
            self.update = self._update_2g
            Hs, Ps = self._Grid_2_EHS()
            if self.normal == False:
                self.get_expectation = self._get_expectation_2g
            elif self.normal == True:
                self.get_expectation = self._get_expectation_2g_normalize
        else:
            raise ValueError
        
        self.h_space = Hs
        self.h_probs = Ps.copy()
        self.initial_h_probs = Ps.copy()

    def _Grid_2_EHS(self):
        num_values =  self.num_values
        Hs = np.linspace(0, 1, self.num_hypothesis).reshape([1,-1]).repeat(axis=0, repeats = num_values).astype(NUMPY_TYPE)
        Ps = np.ones_like(Hs)
        return Hs, Ps
    
    def _get_expectation_2g(self):
        accume_e = (self.h_space * self.h_probs).sum(axis = 1) / self.h_probs.sum(axis = 1)
        return accume_e
    
    def _get_expectation_2g_normalize(self):
        # this version works well when there is no noise
        accume_e = (self.h_space * self.h_probs).sum(axis = 1) / self.h_probs.sum(axis = 1)
        for i in range(self.num_issues):
            range_e_i = self.issue_values_range[i]
            accume_e_i = accume_e[range_e_i[0]:range_e_i[1]]
            max_e_i = accume_e_i.max()
            if max_e_i != 0:
                accume_e[range_e_i[0]:range_e_i[1]] = accume_e_i  / max_e_i
            else:
                accume_e[range_e_i[0]:range_e_i[1]] = np.ones_like(accume_e_i)
        return accume_e

    def update_accume(self):
        self.accume = self.get_expectation()

    def expectation_except_issue_i(self, issue_i):
        h_issue_i = self.h_space[issue_i]
        # prob_issue_i = self.h_probs[issue_i]
        expectation = np.copy(self.accume)
        expectation = expectation.reshape([1,-1]).repeat(repeats=h_issue_i.shape[0], axis = 0)
        value_range = self.issue_values_range[issue_i]
        expectation[:, value_range[0]:value_range[1]] = h_issue_i
        return expectation
    
    def expectation_except_issue_i_value_j_mean(self, faltten_value_j):
        h_issue_i_value_j = self.h_space[faltten_value_j, :]
        # prob_issue_i_value_j = self.h_probs[faltten_value_j, :]
        expectation = np.copy(self.accume)
        expectation = expectation.reshape([1,-1]).repeat(repeats=h_issue_i_value_j.shape[0], axis = 0)
        expectation[:, faltten_value_j] = h_issue_i_value_j
        return expectation

    def expectation_except_issue_i_value_j_rank(self, issue_i, value_j):
        h_issue_i_value_j = self.h_space[issue_i][value_j, :]
        # prob_issue_i_value_j = self.h_probs[issue_i][value_j, :]
        expectation = np.copy(self.accume)
        expectation = expectation.reshape([1,-1]).repeat(repeats=h_issue_i_value_j.shape[0], axis = 0)
        value_range = self.issue_values_range[issue_i]
        value_start = value_range[0]
        value_update = value_start + value_j
        expectation[:, value_update] = h_issue_i_value_j
        return expectation

    def _update_2g(self, accume_w_flatten, likelihood_func):
        for j in range(self.num_values):
            hs_j = self.expectation_except_issue_i_value_j_mean(j)
            ufun_j = hs_j * accume_w_flatten
            Ls = likelihood_func(ufun_j)
            h_probs_j = self.h_probs[j,:] * Ls
            sum_prob = h_probs_j.sum()
            if sum_prob != 0:
                self.h_probs[j,:] = (h_probs_j * h_probs_j.size) / sum_prob
            else:
                self.h_probs[j,:] = np.ones_like(h_probs_j)
            self.update_accume()

    def update_recursive(self, accume_w_flatten, likelihood_func):
        for j in range(self.num_values):
            hs_j = self.expectation_except_issue_i_value_j_mean(j)
            ufun_j = hs_j * accume_w_flatten
            Ls = likelihood_func(ufun_j)
            h_probs_j = self.h_probs[j,:] * Ls
            sum_prob = h_probs_j.sum()
            if sum_prob != 0:
                self.h_probs[j,:] = (h_probs_j * h_probs_j.size) / sum_prob
            else:
                self.h_probs[j,:] = np.ones_like(h_probs_j)
        self.update_accume()

    def resample_inverse(self, gate):
        num_p = self.num_hypothesis
        for j in range(self.num_issues):
            h_j = self.h_space[j, :]
            p_j = self.h_probs[j, :]
            p_j = p_j / p_j.sum()
            N_eff = 1 / np.sum(p_j**2)
            if N_eff >= num_p * gate:
                continue
            # Compute the CDF
            cdf = np.cumsum(p_j)
            uniform_samples = np.linspace(0.05, 1, num_p)

            def inverse_sample(u, cdf, h):
                for i in range(len(cdf)):
                    if u <= cdf[i]:
                        y1 = h[i]
                        if i == len(cdf) - 1:  # If it's the last element
                            y2 = h[i]
                        else:
                            y2 = h[i+1]
                        if i == 0:
                            x1 = 0
                        else:
                            x1 = cdf[i-1]
                        x2 = cdf[i]
                        return y1 + (y2 - y1) * (u - x1) / (x2 - x1)
                return h[-1]
            # Generate samples using inverse sampling
            generated_samples = [inverse_sample(u, cdf, h_j) for u in uniform_samples]
            # print('gs:', generated_samples)
            final_samples = []
            i = 0
            processed = set()  # To keep track of already processed samples
            while i < len(generated_samples):
                sample = generated_samples[i]
                if sample not in processed:
                    count = generated_samples.count(sample)
                    if count == 1:
                        final_samples.append(sample)
                        i += 1
                    else:
                        idx = h_j.tolist().index(sample)
                        start = h_j[idx - 1] if idx - 1 >= 0 else h_j[idx]
                        end = h_j[idx + 1] if idx + 1 < len(h_j) else h_j[idx]
                        step = (end - start) / (count + 1)
                        
                        # Add one occurrence of the original sample
                        final_samples.append(sample)
                        
                        # Interpolate for the remaining samples
                        for _j in range(1, count):
                            final_samples.append(start + step * (_j + 1))
                        
                        i += count  # Skip past all occurrences of the current sample
                        processed.add(sample)  # Mark the sample as processed
                else:
                    i += 1
            # print('fs', final_samples)
            final_samples = np.sort(final_samples)
            
            self.h_space[j, :] = final_samples
            self.h_probs[j, :] = np.ones(self.num_hypothesis)
        

class scalable_OM:

    def __init__(self, ufun = None, opp_ufun = None, e_mode = '2g', w_mode = '1g',
                 e_num = 11, w_num = 11,
                 e_normal = False, w_normal = False,
                 compact_version = None, compact_gate = None,
                 time_max = 5000,
                 SIGMA = 0.15):
        
        self.e_mode = e_mode
        self.w_mode = w_mode
        self.e_num = e_num
        self.w_num = w_num
        self.compact_version = compact_version
        self.compact_gate = compact_gate
        self.e_normal = e_normal # only works when mode = 2
        self.w_normal = w_normal # only works when mode = 1

        self.time_max = time_max
        self.SIGMA = SIGMA

        self.update = self._first_update_func
        self.likelihood_func = self._first_likelihood_func

        if ufun != None:
            self.update_domain(ufun = ufun, opp_ufun = opp_ufun)
        # self.initial()    

    def update_domain(self, ufun, opp_ufun):

        # self.__init__(e_mode = self.e_mode,
        #               w_mode = self.w_mode,
        #               e_num = self.e_num,
        #               w_num = self.w_num,
        #               e_normal = self.e_normal,
        #               w_normal = self.w_normal,
        #               compact_version = self.compact_version,
        #               compact_gate = self.compact_gate,
        #               time_max = self.time_max,
        #               SIGMA = self.SIGMA)
    

        if self.compact_version is not None:
            self.onehot_bids_history_compacted = None
            self.time_sequence_compacted = np.array([], dtype = NUMPY_TYPE)
            if self.compact_version == 'Moving':
                self._compact_bids_times = self._compact_bids_times_Moving
        
        self.update = self._first_update_func
        self.likelihood_func = self._first_likelihood_func
        
        self.opp_ufun = opp_ufun
        
        self.bids_history = []
        self.onehot_bids_history = None
        self.time_sequence = np.array([], dtype = NUMPY_TYPE)

        self.weights = weight_space(ufun = ufun, mode = self.w_mode, normal_part = self.w_normal, num_hypothesis=self.w_num)
        self.evaluations = evaluation_space(ufun = ufun, mode = self.e_mode, normal = self.e_normal, num_hypothesis=self.e_num)

        self.num_outcomes = negmas.outcomes.num_outcomes(ufun.outcome_space.issues)

        self.times_i = 0
        # self.time_correct = 0
        self.first_update = 1

        self.issues = []
        self.values = {}
        self.issue_num = {}
        self.value_num_issue = {} 
        self.issue_value_num_flatten = {}
        self.issueID_valueID_flatten = {}
        self.num_issues = len(ufun.issues)
        _i = 0
        _iii = 0
        self.weights_range = {}
        for k in ufun.issues:
            k_name = k.name
            k_values = k.values
            self.issues.append(k_name)
            self.values[k_name] = k_values
            self.issue_num[k_name] = _i
            self.issue_value_num_flatten[_i] = {}
            self.weights_range[_i] = []
            self.value_num_issue[_i] = {}
            self.issueID_valueID_flatten[_i] = {}
            _ii = 0
            for v in k_values:
                self.value_num_issue[_i][v] = _ii
                self.issueID_valueID_flatten[_i][_ii] = _iii
                _ii = _ii + 1
                self.issue_value_num_flatten[_i][v] = _iii
                self.weights_range[_i].append(_iii)
                _iii = _iii + 1
            _i = _i + 1
        self.num_values = _iii # the lenth of flatten evaluations
        self.update_accumed_ufun()

    def _update_base(self, offer):
        offer_onehot = self.offer_2_onehot(offer)
        self.onehot_bids_history = np.append(self.onehot_bids_history, offer_onehot.reshape([1,-1]), axis = 0)
        self.bids_history.append(offer)
        self.update_time_sequence()
        self._compact_bids_times()

    def _compact_bids_times_Moving(self):
        if self.onehot_bids_history.shape[0] <= self.compact_gate:
            self.onehot_bids_history_compacted = self.onehot_bids_history
            self.time_sequence_compacted = self.time_sequence
        else:
            self.onehot_bids_history_compacted = self.onehot_bids_history[-self.compact_gate:, :]
            self.time_sequence_compacted = self.time_sequence[-self.compact_gate:]
    
    def _compact_bids_times(self):
        pass

    def _update_base_first(self, offer):
        offer_onehot = self.offer_2_onehot(offer)
        self.onehot_bids_history = offer_onehot.reshape([1,-1])

        self.not_proposed = np.ones(self.num_values) * self.num_outcomes / 2
        self.not_proposed = self.not_proposed - offer_onehot
        self.num_unproposed = self.num_outcomes - 1

        self.bids_history.append(offer)
        self.first_update_time_sequence()

    def _update(self, offer, t):
        self._update_base(offer)

        accume_e = self.evaluations.accume.reshape([1,-1])
        self.weights.update(accume_e = accume_e, likelihood_func = self.likelihood_func)

        accume_w_flatten = self.weights.flatten_accume.reshape([1,-1])
        self.evaluations.update(accume_w_flatten = accume_w_flatten, likelihood_func = self.likelihood_func)

        self.update_accumed_ufun()

    def _first_update_func(self, offer, t):
        self._update_base_first(offer)

        accume_e = self.evaluations.accume.reshape([1,-1])
        self.weights.update(accume_e = accume_e, likelihood_func = self.likelihood_func)
        accume_w_flatten = self.weights.flatten_accume.reshape([1,-1])
        self.evaluations.update(accume_w_flatten = accume_w_flatten, likelihood_func = self.likelihood_func)
        
        self.update_accumed_ufun()

        self.evaluations.initial_h_probs = self.evaluations.h_probs
        self.weights.initial_h_probs = self.weights.h_probs
        
        self.first_update = 0

        self.update = self._update
        self.likelihood_func = self._likelihood_func

    def offer_2_onehot(self, offer):
        offer_onehot = np.zeros(self.num_values, dtype='int')
        for i in range(len(offer)):
            value = offer[i]
            loc_value = self.issue_value_num_flatten[i][value]
            offer_onehot[loc_value] = 1
        return offer_onehot
            
    def __call__(self, offer):
        offer_onehot = self.offer_2_onehot(offer)
        util = (self.accume_ufun * offer_onehot).sum(axis = 0)
        return util

    def update_time_sequence(self):
        t_i = self.times_i / self.time_max
        self.time_sequence = np.append(self.time_sequence, t_i)
        self.times_i = self.times_i + 1

    def first_update_time_sequence(self):
        t_i = self.times_i / self.time_max
        self.time_sequence = np.array([t_i])
        self.times_i = self.times_i + 1
    
    def update_accumed_ufun(self):
        evaluations = self.evaluations.accume
        flatten_weights = self.weights.flatten_accume
        self.accume_ufun = flatten_weights * evaluations
    
    @likelihood_none_zero
    def _first_likelihood_func(self, ufuns):
        sigma = self.SIGMA
        first_bid = self.onehot_bids_history[0, :]
        estimated_utilities = (ufuns * first_bid).sum(axis = 1)
        delta = estimated_utilities - 1
        delta[np.where(delta >= 0)] = 0
        likelihood = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(delta * delta) / (2 * sigma * sigma))
        return likelihood


class stable_OM(scalable_OM):

    def _update(self, offer, t):
        pre_prob_e = self.evaluations.h_probs.copy()
        pre_prob_w = self.weights.h_probs.copy()      
        self._update_base(offer)
        ufun_change = 1
        _rep = 0
        while (ufun_change >= 0.001) and (_rep <= 50):
            prev_accumed_ufun = self.accume_ufun
            self.evaluations.h_probs = pre_prob_e # this step set the probabilities to the initial state of this step
            self.weights.h_probs = pre_prob_w # while not set the accumelated utility function to the initial state
            # print(ufun_change)
            _rep = _rep + 1
            accume_e = self.evaluations.accume.reshape([1,-1])
            accume_w_flatten = self.weights.flatten_accume.reshape([1,-1])

            self.weights.update_recursive(accume_e = accume_e, likelihood_func = self.likelihood_func)
            self.evaluations.update_recursive(accume_w_flatten = accume_w_flatten, likelihood_func = self.likelihood_func)

            # self.weights.update_accume()
            # self.evaluations.update_accume()

            self.update_accumed_ufun()
            ufun_change = np.abs(prev_accumed_ufun - self.accume_ufun).sum() / self.num_issues

    def _first_update_func(self, offer, t):
        pre_prob_e = self.evaluations.h_probs.copy()
        pre_prob_w = self.weights.h_probs.copy()
        self._update_base_first(offer)
        ufun_change = 1
        _rep = 0
        while (ufun_change >= 0.001) and (_rep <= 50):
            prev_accumed_ufun = self.accume_ufun
            self.evaluations.h_probs = pre_prob_e
            self.weights.h_probs = pre_prob_w
            # print(ufun_change)
            _rep = _rep + 1
            accume_e = self.evaluations.accume.reshape([1,-1])
            accume_w_flatten = self.weights.flatten_accume.reshape([1,-1])
            
            self.weights.update_recursive(accume_e = accume_e, likelihood_func = self.likelihood_func)
            self.evaluations.update_recursive(accume_w_flatten = accume_w_flatten, likelihood_func = self.likelihood_func)

            # self.weights.update_accume()
            # self.evaluations.update_accume()

            self.update_accumed_ufun()
            ufun_change = np.abs(prev_accumed_ufun - self.accume_ufun).sum() / self.num_issues

        self.evaluations.initial_h_probs = self.evaluations.h_probs
        self.weights.initial_h_probs = self.weights.h_probs
        
        self.first_update = 0

        self.update = self._update
        self.likelihood_func = self._likelihood_func

class resample_OM(scalable_OM):

    def __init__(self, resample_gate, resample_mode, ufun=None, opp_ufun=None, e_mode='2g', w_mode='1g', e_num=11, w_num=11, e_normal=False, w_normal=False, compact_version=None, compact_gate=None, time_max=5000, SIGMA=0.15):
        super().__init__(ufun, opp_ufun, e_mode, w_mode, e_num, w_num, e_normal, w_normal, compact_version, compact_gate, time_max, SIGMA)
        self.resample_gate = resample_gate
        self.resample_mode = resample_mode

    def update_domain(self, ufun, opp_ufun):
        super().update_domain(ufun, opp_ufun)
        # if self.resample_mode == 'inverse':
        self.weights.resample = self.weights.resample_inverse
        self.evaluations.resample = self.evaluations.resample_inverse
        print(self.resample_mode, self.resample_gate)

    def _update(self, offer, t):
        self._update_base(offer)

        accume_e = self.evaluations.accume.reshape([1,-1])
        self.weights.update(accume_e = accume_e, likelihood_func = self.likelihood_func)

        accume_w_flatten = self.weights.flatten_accume.reshape([1,-1])
        self.evaluations.update(accume_w_flatten = accume_w_flatten, likelihood_func = self.likelihood_func)

        self.update_accumed_ufun()

        self.weights.resample(self.resample_gate)
        self.evaluations.resample(self.resample_gate)

    def _first_update_func(self, offer, t):
        self._update_base_first(offer)

        accume_e = self.evaluations.accume.reshape([1,-1])
        self.weights.update(accume_e = accume_e, likelihood_func = self.likelihood_func)
        accume_w_flatten = self.weights.flatten_accume.reshape([1,-1])
        self.evaluations.update(accume_w_flatten = accume_w_flatten, likelihood_func = self.likelihood_func)
        
        self.update_accumed_ufun()

        self.weights.resample(self.resample_gate)
        self.evaluations.resample(self.resample_gate)        

        self.evaluations.initial_h_probs = self.evaluations.h_probs
        self.weights.initial_h_probs = self.weights.h_probs
        
        self.first_update = 0

        self.update = self._update
        self.likelihood_func = self._likelihood_func

class stable_resample(resample_OM):

    def _update(self, offer, t):
        pre_prob_e = self.evaluations.h_probs.copy()
        pre_prob_w = self.weights.h_probs.copy()      
        self._update_base(offer)
        ufun_change = 1
        _rep = 0
        while (ufun_change >= 0.001) and (_rep <= 50):
            prev_accumed_ufun = self.accume_ufun
            self.evaluations.h_probs = pre_prob_e
            self.weights.h_probs = pre_prob_w
            # print(ufun_change)
            _rep = _rep + 1
            accume_e = self.evaluations.accume.reshape([1,-1])
            accume_w_flatten = self.weights.flatten_accume.reshape([1,-1])

            self.weights.update_recursive(accume_e = accume_e, likelihood_func = self.likelihood_func)
            self.evaluations.update_recursive(accume_w_flatten = accume_w_flatten, likelihood_func = self.likelihood_func)

            self.update_accumed_ufun()
            ufun_change = np.abs(prev_accumed_ufun - self.accume_ufun).sum() / self.num_issues

        self.weights.resample(self.resample_gate)
        self.evaluations.resample(self.resample_gate)
    
    def _first_update_func(self, offer, t):
        pre_prob_e = self.evaluations.h_probs.copy()
        pre_prob_w = self.weights.h_probs.copy()
        self._update_base_first(offer)
        ufun_change = 1
        _rep = 0
        while (ufun_change >= 0.001) and (_rep <= 50):
            prev_accumed_ufun = self.accume_ufun
            self.evaluations.h_probs = pre_prob_e
            self.weights.h_probs = pre_prob_w
            # print(ufun_change)
            _rep = _rep + 1
            accume_e = self.evaluations.accume.reshape([1,-1])
            accume_w_flatten = self.weights.flatten_accume.reshape([1,-1])
            
            self.weights.update_recursive(accume_e = accume_e, likelihood_func = self.likelihood_func)
            self.evaluations.update_recursive(accume_w_flatten = accume_w_flatten, likelihood_func = self.likelihood_func)

            self.weights.resample(self.resample_gate)
            self.evaluations.resample(self.resample_gate)

            self.update_accumed_ufun()
            ufun_change = np.abs(prev_accumed_ufun - self.accume_ufun).sum() / self.num_issues

        self.evaluations.initial_h_probs = self.evaluations.h_probs
        self.weights.initial_h_probs = self.weights.h_probs
        
        self.first_update = 0

        self.update = self._update
        self.likelihood_func = self._likelihood_func
