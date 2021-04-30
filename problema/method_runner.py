import numpy as np

from scipy import stats as stts

from problema.clustering import Clustering
from .utils import evaluate_dists_state
from heuristics import grasp, simulated_annealing, genetic

from sklearn.cluster import KMeans

import time as time_




class MethodRunner:
    def __init__(self, hparams, time = 1.):
        self.hparams = hparams
        self.t = time
        # self.result = self.init_result()

    def run(self, problem, k, hparam):
        raise Exception("NotImplementedException")

    # list by dict
    def run_problem(self, problem: Clustering, ks, times):
        r = {k: {} for k in ks}
        for k in r:
            r[k]['sse'] = {}
            r[k]['t'] = {}
            for i, param in enumerate(self.hparams):
                run_results = [self.run(problem, k, param) for _ in range(times)]
                print(k, param)
                sse_mean, time_mean = np.mean(run_results, axis=0)
                r[k]['sse'][i] = sse_mean
                r[k]['t'][i] = time_mean
            sses = list(r[k]['sse'].values())
            r[k]['zscore'] = np.nan_to_num(stts.zscore(sses))
            r[k]['zscore'] = dict(enumerate(r[k]['zscore']))
            r[k]['rank'] = dict(enumerate(stts.rankdata(sses)))
        return r


class GraspRunner(MethodRunner):
    def run(self, problem, k, hparam):
        _, _, t, _, min_dists = grasp(problem, k, hparam['n_best'], hparam['n_iter'], self.t)
        return (evaluate_dists_state(min_dists), t)

class SARunner(MethodRunner):
    def run(self, problem, k, hparam):
        centroids, t, = simulated_annealing(problem, k, hparam['t_zero'], hparam['alfa'], 
                                            hparam["n_iter"], min_t = 0.01,tempo = self.t)
        _, min_dists = problem.assign_clusters(centroids)
        return (evaluate_dists_state(min_dists), t)

class GeneticRunner(MethodRunner):
    def run(self, problem, k, hparam):
        centroids, t, _ = genetic(problem, k, hparam['t_pop'], hparam['t_cross'], hparam['t_mut'], self.t)
        _, min_dists = problem.assign_clusters(centroids)
        return (evaluate_dists_state(min_dists), t)

class KmeansRunner(MethodRunner):
    def run(self, problem, k, hparam=None):
        start = time_.process_time()
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(problem.data)

        _, min_dists = problem.assign_clusters(kmeans.cluster_centers_)
        return (evaluate_dists_state(min_dists), time_.process_time() - start)

    # def run_problem(self, problem: Clustering, ks, times):
    #     r = {k: {} for k in ks}
    #     for k in r:
    #         r[k]['sse'] = {}
    #         r[k]['t'] = {}

    #         run_results = [self.run(problem, k) for _ in range(times)]
    #         print(k)

    #         sse_mean, time_mean = np.mean(run_results, axis=0)
    #         r[k]['sse'][0] = sse_mean
    #         r[k]['t'][0] = time_mean

    #         sses = list(r[k]['sse'].values())
    #         r[k]['zscore'] = np.nan_to_num(stts.zscore(sses))
    #         r[k]['zscore'] = dict(enumerate(r[k]['zscore']))
    #         r[k]['rank'] = dict(enumerate(stts.rankdata(sses)))
    #     return r