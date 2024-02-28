from abc import ABCMeta, abstractmethod
import numpy as np
from copy import deepcopy
import itertools
import math

class AbstractOpponentModel(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, offer) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, offer, t) -> None:
        raise NotImplementedError


# No knowledge about the preference profile
class NoModel(AbstractOpponentModel):
    def __call__(self, offer):
        pass

    def update(self, offer, t):
        pass

# Learns the issue weights based on how often the value of an issue changes
# The value weights are estimated based on the frequency they are offered
class HardHeadedFrequencyModel(AbstractOpponentModel):
    
    def __init__(self, ufun, learn_coef=0.2, learn_value_addition=1):
        self.weights = {}
        self.evaluates = {}
#     issue_names = []
        self.prevOffer = None
        self.amountOfIssues = len(ufun.issues)
        self.learnCoef = learn_coef
        self.learnValueAddition = learn_value_addition
        self.gamma = 0.25
        self.goldenValue = self.learnCoef / self.amountOfIssues
        for k in ufun.issues:
            k_name = k.name
            k_values = k.values
            self.weights[k_name] = (1.0 / self.amountOfIssues)
#             self.issue_names.append(k_name)
            self.evaluates[k_name] = {i: 1.0 for i in k.values}
    
    def __call__(self, offer):
        util = 0
        weights = list(self.weights.values())
        issues = list(self.weights.keys())
        for w, i, v in zip(weights, issues, offer):
            try:
                util += w * (self.evaluates[i][v] / max(self.evaluates[i].values()))
            except:
                print(v)
                print(self.evaluates[i])
                raise KeyError('evalue')
        return util

    def update(self, offer, t):
        if self.prevOffer is not None:
            last_diff = self.determine_difference(offer, self.prevOffer)
            num_of_unchanged = len(last_diff) - sum(last_diff)
            total_sum = 1 + self.goldenValue * num_of_unchanged
            maximum_weight = 1 - self.amountOfIssues * self.goldenValue / total_sum
            
            for k, i in zip(self.weights.keys(), last_diff):
                weight = self.weights[k]
                if i == 0 and weight < maximum_weight:
                    self.weights[k] = (weight + self.goldenValue) / total_sum
                else:
                    self.weights[k] = weight / total_sum

        for issue, evaluator in zip(self.weights.keys(), offer):
            self.evaluates[issue][evaluator] += self.learnValueAddition
        self.prevOffer = offer

    @staticmethod
    def determine_difference(first, second):
        return [int(f == s) for f, s in zip(first, second)]

# Counts how often each value is offered
# The utility of a bid is the sum of the score of its values divided by the best possible score
# The model only uses the first 100 unique bids for its estimation
class CUHKAgentValueModel(AbstractOpponentModel):

    def __init__(self, ufun):
        self.evaluates = []
        self.bid_history = []
        self.maximumBidsStored = 100
        self.maxPossibleTotal = 0
        for j in range(len(ufun.issues)):
            # k = ufun.issues[j]
            self.evaluates.append({i: 0.0 for i in ufun.issues[j].values})

    def __call__(self, offer):
        total_bid_value = 0.0
        for i in range(len(offer)):
            v = offer[i]
            counter_per_value = self.evaluates[i][v]
            total_bid_value += counter_per_value
        if total_bid_value == 0:
            return 0.0
        return total_bid_value / self.maxPossibleTotal

    def update(self, offer, t):
        if offer not in self.bid_history and len(self.bid_history) < self.maximumBidsStored:
            self.bid_history.append(offer)
            for i in range(len(offer)):
                v = offer[i]
                if self.evaluates[i][v] + 1 > max(self.evaluates[i].values()):
                    self.maxPossibleTotal += 1
                self.evaluates[i][v] += 1

# Defines the opponent’s utility as one minus the agent’s utility
class OppositeModel(AbstractOpponentModel):
    def __init__(self, my_ufun):
        self.ufun = my_ufun

    def __call__(self, offer):
        return 1. - self.ufun(offer)

    def update(self, offer, t):
        pass

# Perfect knowledge of the opponent’s preferences
class PerfectModel(AbstractOpponentModel):
    def __init__(self, opp_ufun):
        self.ufun = opp_ufun

    def __call__(self, offer):
        return self.ufun(offer)

    def update(self, offer, t):
        pass

# Defines the estimated utility as one minus the real utility
class WorstModel(AbstractOpponentModel):
    def __init__(self, opp_ufun):
        self.ufun = opp_ufun

    def __call__(self, offer):
        return 1. - self.ufun(offer)

    def update(self, offer, t):
        pass

