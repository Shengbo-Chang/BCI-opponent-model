
import numpy as np
from negmas.preferences import pareto_frontier
from shapely.geometry import Polygon
import time as s_time
from func_timeout import func_timeout, exceptions

class history_update():
    
    def Multi_OM_update_from_history(self, ins_opp_models, history):
        evalues_all = []
        t_max = len(history)
        for t_n in range(t_max):
            if t_max != 1:
                t = t_n / (t_max - 1)
            else:
                t = 0
            bid = history[t_n]
            evalues = []
            print(t_n)
            for m in ins_opp_models:
                m.update(bid, t)
                evalues.append(self.evalue(m))
            evalues_all.append(evalues)
            print(evalues)
        return evalues_all
    
    def Multi_OM_update_from_history_10(self, ins_opp_models, history):
        evalues_all = []
        t_max = len(history)
        for t_n in range(t_max):
            if t_max != 1:
                t = t_n / (t_max - 1)
            else:
                t = 0
            bid = history[t_n]
            evalues = []
            for m_i in range(len(ins_opp_models)):
                m = ins_opp_models[m_i]
                m.update(bid, t)
                if t_n % 10 == 0 or t_n == (t_max - 1):
                    evalue = self.evalue(m)
                    if np.isnan(evalue):
                        try:
                            evalue = evalues_all[-1][m_i]
                        except:
                            evalue = 0
                    evalues.append(evalue)
            if t_n % 10 == 0 or t_n == (t_max - 1):
                evalues_all.append(evalues)
                print(t_n)
                # print(evalues)
        return evalues_all
    
    def time_cost_calculate(self, history):
        evalues = []
        t_max = len(history)
        m = self.ins_opp_models[0]
        for t_n in range(t_max):
            t = t_n / (t_max - 1)
            bid = history[t_n]
            st_0 = s_time.time()
            m.update(bid, t)
            st_1 = s_time.time()
            evalues.append(st_1 - st_0)
        evalues.append(evalues)
        return evalues

    def single_OM_update_from_history_10(self, opp_model, history):
        evalues = []
        t_max = len(history)
        try:
            for t_n in range(t_max):
                if t_max != 1:
                    t = t_n / (t_max - 1)
                else:
                    t = 0
                bid = history[t_n]
                opp_model.update(bid, t)
                if t_n % 10 == 0 or t_n == (t_max - 1):
                    print(t_n)
                    evalue = self.evalue(opp_model)
                    if np.isnan(evalue):
                        try:
                            evalue = evalues[-1]
                        except:
                            evalue = 0
                    evalues.append(evalue)
            return evalues
        except exceptions.FunctionTimedOut:
            return 'Time_out'
    
    def single_OM_update_from_history(self, opp_model, history):
        evalues = []
        t_max = len(history)
        try:
            for t_n in range(t_max):
                if t_max != 1:
                    t = t_n / (t_max - 1)
                else:
                    t = 0
                bid = history[t_n]
                # print(t_n)
                # func_timeout(5, opp_model.update, args=(bid, t))
                opp_model.update(bid, t)
                print(t_n)
                evalue = self.evalue(opp_model)
                print(evalue)
                evalues.append(evalue)
            return evalues
        except exceptions.FunctionTimedOut:
            return 'Time_out'

class evaluation_matrix_pearson(history_update):
    
    def __init__(self, ufun_opp, ufun_own, outcomes):
        self.ufun_opp = ufun_opp
        self.outcomes = outcomes
        self.ufun_own = ufun_own
        self.outcomes_u = np.array(list(map(ufun_opp, outcomes))).astype('float32')
        self.outcomes_u_own = np.array(list(map(ufun_own, outcomes))).astype('float32')
        self.outcomes_real_us = np.append(self.outcomes_u_own.reshape([-1,1]), self.outcomes_u.reshape([-1,1]), axis = 1)

    def evalue(self, e_ufun):
        e_outcomes_u = np.array(list(map(e_ufun, self.outcomes))).astype('float32')
        return self._pearson_correlation(e_outcomes_u)
    
    def _pearson_correlation(self, e_outcomes_u):
        diff_e = e_outcomes_u - e_outcomes_u.mean() 
        diff_r = self.outcomes_u - self.outcomes_u.mean()
        if diff_e.all() != 0:
            pearson = np.sum(diff_e * diff_r) / np.sqrt(np.sum(diff_e**2)*np.sum(diff_r**2))
        else:
            pearson = 'Nan'
        return float(pearson)

class evaluation_matrix_pareto(history_update):
    
    def __init__(self, ufun_opp, ufun_own, outcomes):
        self.ufun_opp = ufun_opp
        self.outcomes = outcomes
        self.ufun_own = ufun_own
        self.outcomes_u = np.array(list(map(ufun_opp, outcomes))).astype('float32')
        self.outcomes_u_own = np.array(list(map(ufun_own, outcomes))).astype('float32')
        self.outcomes_real_us = np.append(self.outcomes_u_own.reshape([-1,1]), self.outcomes_u.reshape([-1,1]), axis = 1)
        self.r_pareto = None
        self.r_pareto_u = None
        self.r_pareto_surface = None
        self._update_r_pareto()
        
    def _is_pareto_efficient(self, costs, return_mask = False):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0 # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask] # Remove dominate
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient
        
    def _area_calculate(self, pareto_u):
        poly_p = Polygon(pareto_u)
        return poly_p.area
    
    def _update_r_pareto(self):
        r_pareto_u, r_pareto = pareto_frontier([self.ufun_own, self.ufun_opp], self.outcomes)
        r_pareto_u.append([0,1])
        r_pareto_u.append([0,0])
        r_pareto_u = [[1,0]] + r_pareto_u
        r_pareto_u = np.array(r_pareto_u)
        self.r_pareto_area = Polygon(r_pareto_u).area
        # print(r_pareto_u)
    
    def evalue(self, e_ufun):
        e_outcomes_u = np.array(list(map(e_ufun, self.outcomes)))
        outcomes_e_us = np.append(self.outcomes_u_own.reshape([-1,1]), e_outcomes_u.reshape([-1,1]), axis = 1).astype('float32')
        e_pareto_i = self._is_pareto_efficient(- outcomes_e_us)
        e_pareto_u = self.outcomes_real_us[e_pareto_i]
        e_pareto_u = e_pareto_u[e_pareto_u[:,0].argsort()][::-1]
        e_pareto_u = np.append(e_pareto_u, [[0,e_pareto_u[-1,1]]], axis=0)
        e_pareto_u = np.append(e_pareto_u, [[0,0]], axis=0)
        e_pareto_u = np.insert(e_pareto_u, 0, [[1,0]], axis=0)
        e_pareto_area = self._area_calculate(e_pareto_u)
        return float(self.r_pareto_area - e_pareto_area)



    
