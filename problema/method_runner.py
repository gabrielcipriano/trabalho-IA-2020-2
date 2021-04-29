import numpy as np
import itertools

from scipy import stats as stts

from problema.clustering import Clustering
from .utils import evaluate_dists_state
from heuristics import grasp, simulated_annealing, genetic




class MethodRunner:
    def __init__(self, problem: Clustering, hparams, ks, time = 1.):
        self.hparams = hparams
        self.problem = problem
        self.ks = ks
        self.t = time
        self.result = {k: {} for k in ks}
        # self.result = self.init_result()

    def run(self, k, hparam):
        raise Exception("NotImplementedException")

    # list by dict
    def run_all(self, times):
        r = self.result
        for k in r:
            r[k]['sse'] = {}
            r[k]['t'] = {}
            for i, param in enumerate(self.hparams):
                run_results = [self.run(k, param) for _ in range(times)]
                # print(k, hparam, run_results)
                sse_mean, time_mean = np.mean(run_results, axis=0)
                r[k]['sse'][i] = sse_mean
                r[k]['t'][i] = time_mean
            sses = list(r[k]['sse'].values())
            r[k]['zscore'] = np.nan_to_num(stts.zscore(sses))
            r[k]['zscore'] = dict(enumerate(r[k]['zscore']))
            r[k]['rank'] = dict(enumerate(stts.rankdata(sses)))


class GraspRunner(MethodRunner):
    def run(self, k, hparam):
        _, _, t, _, min_dists = grasp(self.problem, k, hparam['n_best'], hparam['n_iter'], self.t)
        return (evaluate_dists_state(min_dists), t)

class SARunner(MethodRunner):
    def run(self, k, hparam):
        centroids, t, = simulated_annealing(self.problem, k, hparam['t_zero'], hparam['alfa'], 
                                            hparam["n_iter"], min_t = 0.01,tempo = self.t)
        _, min_dists = self.problem.assign_clusters(centroids)
        return (evaluate_dists_state(min_dists), t)

class GeneticRunner(MethodRunner):
    def run(self, k, hparam):
        centroids, t, _ = genetic(self.problem, k, hparam['t_pop'], hparam['t_cross'], hparam['t_mut'], self.t)
        _, min_dists = self.problem.assign_clusters(centroids)
        return (evaluate_dists_state(min_dists), t)