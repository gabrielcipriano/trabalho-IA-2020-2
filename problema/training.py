import numpy as np
import itertools

from scipy import stats as stts

from problema.clustering import Clustering
from .utils import evaluate_dists_state
from heuristics import grasp, simulated_annealing, genetic




class Training:
    def __init__(self, problem: Clustering, hparams, ks, time = 1.):
        self.hparams = self.cartesian_product(hparams)
        self.problem = problem
        self.ks = ks
        self.t = time
        self.results_by_k = {k: {} for k in ks}
        # self.result = self.init_result()

    def cartesian_product(self, hparams):
        raise Exception("NotImplementedException")

    def run(self, k, hparam):
        raise Exception("NotImplementedException")

    # def init_result(self):
    #     result = {}
    #     result['sse_mean'] = {}
    #     result['t_mean'] = {}
    #     result['z_score'] = {}
    #     result['rank'] = {}
    #     return result

    def get_result(self):
        result = {}
        result["results_by_k"] = self.results_by_k
        result["hparams"] = self.hparams
        return result

    def train(self, times):
        r = self.results_by_k
        for k in self.ks:
            r[k]['sse_mean'] = []
            r[k]['t_mean'] = []
            for hparam in self.hparams:
                run_results = [self.run(k, hparam) for _ in range(times)]
                print(k, hparam, run_results)
                sse_mean, time_mean = np.mean(run_results, axis=0)
                r[k]['sse_mean'].append(sse_mean)
                r[k]['t_mean'].append(time_mean)
            r[k]['sse_mean'] = np.asarray(r[k]['sse_mean'])
            r[k]['t_mean'] = np.asarray(r[k]['t_mean'])
            r[k]['z_score'] = np.nan_to_num(stts.zscore(r[k]['sse_mean']))
            r[k]['rank'] = stts.rankdata(r[k]['sse_mean'])

class TrainGrasp(Training):
    def cartesian_product(self, hparams):
        products = list(itertools.product(hparams[0], hparams[1]))
        return np.asarray(products, dtype={'names':('n_iter', 'n_best'),'formats':('i4', 'i4')})

    def run(self, k, hparam):
        _, _, t, _, min_dists = grasp(self.problem, k, hparam['n_best'], hparam['n_iter'], self.t)
        return (evaluate_dists_state(min_dists), t)

class TrainSA(Training):
    def cartesian_product(self, hparams):
        products = list(itertools.product(hparams[0], hparams[1], hparams[2]))
        return np.asarray(products, dtype={'names':('t_zero', 'alfa', 'n_iter'),'formats':('f4', 'f4', 'i4')})

    def run(self, k, hparam):
        centroids, t, = simulated_annealing(self.problem, k, hparam['t_zero'], hparam['alfa'], 
                                            hparam["n_iter"], min_t = 0.01,tempo = self.t)
        _, min_dists = self.problem.assign_clusters(centroids)
        return (evaluate_dists_state(min_dists), t)

class TrainGenetic(Training):
    def cartesian_product(self, hparams):
        products = list(itertools.product(hparams[0], hparams[1], hparams[2]))
        return np.asarray(products, dtype={'names':('t_pop', 't_cross', 't_mut'),'formats':('i4', 'f4', 'f4')})

    def run(self, k, hparam):
        centroids, t, _ = genetic(self.problem, k, hparam['t_pop'], hparam['t_cross'], hparam['t_mut'], self.t)
        _, min_dists = self.problem.assign_clusters(centroids)
        return (evaluate_dists_state(min_dists), t)