from .base import scalable_OM, likelihood_none_zero, resample_OM, stable_OM, resample_stable
# import , likelihood_none_zero
import numpy as np


class BCI_no_both(scalable_OM):
    def __init__(self, ufun=None, opp_ufun=None, e_mode='2g', w_mode='1g', e_num=11, w_num=11, e_normal=False, w_normal=False, compact_version=None, compact_gate=None, time_max=5000, SIGMA=0.15):
        super().__init__(ufun, opp_ufun, e_mode, w_mode, e_num, w_num, e_normal, w_normal, compact_version, compact_gate, time_max, SIGMA)

    @likelihood_none_zero
    def _likelihood_func(self, ufuns):
        sigma = self.SIGMA
        bids_till_now = self.onehot_bids_history

        previous_bid = bids_till_now[-2, :]
        newest_bid = bids_till_now[-1, :]

        diff_bid = newest_bid - previous_bid
        SIM_single_step = (ufuns * diff_bid).sum(axis = 1)

        # SIM_single_step[np.where((-0.2 <= SIM_single_step) & (SIM_single_step <= 0))] = 0
        SIM_single_step[np.where((-0.1 <= SIM_single_step) & (SIM_single_step <= 0))] = 0
        likelihood_single = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(SIM_single_step * SIM_single_step) / (2 * sigma * sigma))
        return likelihood_single
    
class BCI_no_interdependence(resample_OM):

    def __init__(self, resample_gate=0.5, ufun=None, opp_ufun=None, e_mode='2g', w_mode='1g', e_num=11, w_num=11, e_normal=False, w_normal=True, compact_version=None, compact_gate=None, time_max=5000, SIGMA=0.15):
        super().__init__(ufun, opp_ufun, e_mode, w_mode, e_num, w_num, e_normal, w_normal, compact_version, compact_gate, time_max, SIGMA)
        self.resample_gate = resample_gate

    @likelihood_none_zero
    def _likelihood_func(self, ufuns):
        sigma = self.SIGMA
        bids_till_now = self.onehot_bids_history

        previous_bid = bids_till_now[-2, :]
        newest_bid = bids_till_now[-1, :]

        diff_bid = newest_bid - previous_bid
        SIM_single_step = (ufuns * diff_bid).sum(axis = 1)

        # SIM_single_step[np.where((-0.2 <= SIM_single_step) & (SIM_single_step <= 0))] = 0
        SIM_single_step[np.where((-0.1 <= SIM_single_step) & (SIM_single_step <= 0))] = 0
        likelihood_single = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(SIM_single_step * SIM_single_step) / (2 * sigma * sigma))
        return likelihood_single

class BCI_no_resampling(stable_OM):

    def __init__(self, ufun=None, opp_ufun=None, e_mode='2g', w_mode='1g', e_num=11, w_num=11, e_normal=True, w_normal=False, compact_version=None, compact_gate=None, time_max=5000, SIGMA=0.15):
        super().__init__(ufun, opp_ufun, e_mode, w_mode, e_num, w_num, e_normal, w_normal, compact_version, compact_gate, time_max, SIGMA)

    @likelihood_none_zero
    def _likelihood_func(self, ufuns):
        sigma = self.SIGMA
        bids_till_now = self.onehot_bids_history

        previous_bid = bids_till_now[-2, :]
        newest_bid = bids_till_now[-1, :]

        diff_bid = newest_bid - previous_bid
        SIM_single_step = (ufuns * diff_bid).sum(axis = 1)

        # SIM_single_step[np.where((-0.2 <= SIM_single_step) & (SIM_single_step <= 0))] = 0
        SIM_single_step[np.where((-0.1 <= SIM_single_step) & (SIM_single_step <= 0))] = 0
        likelihood_single = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(SIM_single_step * SIM_single_step) / (2 * sigma * sigma))
        return likelihood_single
    
class BCI(resample_stable):

    def __init__(self, resample_gate=0.5, ufun=None, opp_ufun=None, e_mode='2g', w_mode='1g', e_num=11, w_num=11, e_normal=True, w_normal=False, compact_version=None, compact_gate=None, time_max=5000, SIGMA=0.15):
        super().__init__(ufun, opp_ufun, e_mode, w_mode, e_num, w_num, e_normal, w_normal, compact_version, compact_gate, time_max, SIGMA)
        self.resample_gate = resample_gate

    @likelihood_none_zero
    def _likelihood_func(self, ufuns):
        sigma = self.SIGMA
        bids_till_now = self.onehot_bids_history

        previous_bid = bids_till_now[-2, :]
        newest_bid = bids_till_now[-1, :]

        diff_bid = newest_bid - previous_bid
        SIM_single_step = (ufuns * diff_bid).sum(axis = 1)

        # SIM_single_step[np.where((-0.2 <= SIM_single_step) & (SIM_single_step <= 0))] = 0
        SIM_single_step[np.where((-0.1 <= SIM_single_step) & (SIM_single_step <= 0))] = 0
        likelihood_single = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(SIM_single_step * SIM_single_step) / (2 * sigma * sigma))
        return likelihood_single
    
