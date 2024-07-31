#!/usr/bin/env python3

import numpy as np
import bottleneck as bn
from scipy.special import gamma, gammaln
from scipy.stats import beta, truncnorm, mode
from scipy.stats import gamma as gamma_fct


np.seterr(divide='raise', over='ignore', under='ignore', invalid= 'raise')
EPSILON = np.finfo(np.float64).resolution
TMIN = 1e-5
TMAX = 1 - TMIN
log_EPSILON = np.log(EPSILON)


class CRP:
    """
    Arguments:
        data (np.array): n x m matrix with n cells and m mutations
            containing 0|1|np.nan
        alpha (float): Concentration Parameter for the CRP
        param_beta ((float, float)): Beta dist parameters used as parameter prior
        FN_error (float): Fixed false negative rate
        FP_error (float): Fixed false positive rate
    """
    def __init__(self, data, timepoint_x, DP_alpha=np.array([-1, -1]), param_beta=np.array([1, 1]), FN_error=EPSILON,
                FP_error=EPSILON, Miss_error=EPSILON, num_times=1):
        # Fixed data
        self.data = data
        self.timepoint_x = timepoint_x
        self.num_times = num_times
        self.cells_total, self.muts_total = self.data.shape

        # Cluster parameter prior (beta function) parameters
        self.p, self.q = param_beta
        self.param_prior = beta(self.p, self.q)

        #if self.p == self.q == 1:
        #Bhavya Changes here // checks if all elements of array are equal to 1
        if np.all(param_beta == 1):
            self.beta_prior_uniform = True
        else:
            self.beta_prior_uniform = False

        mix0 = self.beta_fct(self.p, self.q + 1)
        mix1 = self.beta_fct(self.p + 1, self.q)
        self._beta_mix_const = np.array([mix0, mix1]) / (mix0 + mix1)

        # Error rates
        # 20240625 Liting: Add missing rate and make errors as arrays with size of numTimes
        self.FP = FP_error
        self.FN = FN_error
        self.Miss = Miss_error
        if len(FP_error) != num_times:
            self.FP = FP_error * num_times
        if len(FN_error) != num_times:
            self.FN = FN_error * num_times
        if len(FP_error) != num_times:
            self.Miss = Miss_error * num_times

        # DP alpha
        #if DP_alpha[0] < 0 or DP_alpha[1] < 0:
        #Bhavya Changes here // checks if any element in DP_alpha is less than 0
        if isinstance(DP_alpha, list):
            DP_alpha = np.array(DP_alpha)

        if np.any(DP_alpha < 0):
            self.DP_a_gamma = (np.sqrt(self.cells_total), 1)
        else:
            self.DP_a_gamma = DP_alpha
        self.DP_a_prior = gamma_fct(*self.DP_a_gamma)
        self.DP_a = self.DP_a_prior.mean()

        # Flexible data - Initialization
        self.CRP_prior = None
        self.assignment = None
        self.parameters = None
        self.cells_per_cluster = None #{cluster_id: number of cells in this cluster}

        # MH proposal stDev's
        self.param_proposal_sd = np.array([0.1, 0.25, 0.5])


    def __str__(self):
        out_str = '\nDPMM with:\n' \
            f'\t{self.cells_total} cells\n\t{self.muts_total} mutations\n' \
            f'\tFixed FN rate: {self.FN}\n\tFixed FP rate: {self.FP}\n' \
            '\n\tPriors:\n' \
            f'\tGroud Truth Params.:\tBeta({self.p},{self.q})\n' \
            f'\tCRP a_0:\tGamma({self.DP_a_gamma[0]:.1f},{self.DP_a_gamma[1]})\n'
        return out_str


    @staticmethod
    def beta_fct(p, q):
        return gamma(p) * gamma(q) / gamma(p + q)


    @staticmethod
    def log_CRP_prior(n_i, n, a, dtype=np.float64):
        return np.log(n_i, dtype=dtype) - np.log(n - 1 + a, dtype=dtype)

    def get_sci(self):
        # get singleton cluster index
        # Count the total number of clusters
        total_clusters = len(self.cells_per_cluster)
        # Count the number of singleton clusters
        singleton_clusters = sum(1 for count in self.cells_per_cluster.values() if count == 1)
        psc = 0 # percent of singleton cluster
        # Calculate the percentage of singleton clusters
        if total_clusters != 0:
            psc = (singleton_clusters / total_clusters) 
        # percent of cells in singleton cluster
        #pcc = singleton_clusters / self.cells_total
        return psc
        #return 0.6 * psc + 0.4 * pcc
    
    @staticmethod
    def _normalize_log_probs(probs):
        max_i = bn.nanargmax(probs)
        try:
            exp_probs = np.exp(probs[np.arange(probs.size) != max_i] \
                - probs[max_i])
        except FloatingPointError:
            exp_probs = np.exp(
                np.clip(probs[np.arange(probs.size) != max_i] - probs[max_i],
                    log_EPSILON, 0)
            )
        probs_norm = probs - probs[max_i] - np.log1p(bn.nansum(exp_probs))
        return np.exp(np.clip(probs_norm, log_EPSILON, 0))


    @staticmethod
    def _normalize_log(probs):
        max_i = bn.nanargmax(probs, axis=0)
        try:
            log_probs_norm = probs - probs[max_i] - np.log1p(bn.nansum(
                np.exp(probs[np.arange(probs.size) != max_i] - probs[max_i])
            ))
        except FloatingPointError:
            if probs[0] > probs[1]:
                return np.array([0, log_EPSILON])
            else:
                return np.array([log_EPSILON, 0])
        else:
            return log_probs_norm

      
    def init(self, mode='random', assign=False):
        # Predefined assignment vector
        if assign:
            self.assignment = np.array(assign)
            self.cells_per_cluster = {}
            cl, cl_size = np.unique(assign, return_counts=True)
            for i in range(cl.size):
                bn.replace(self.assignment, cl[i], i)
                self.cells_per_cluster[i] = cl_size[i]
            self.parameters = self._init_cl_params('assign')
        elif mode == 'separate':
            self.assignment = np.arange(self.cells_total, dtype=int)
            self.cells_per_cluster = {i: 1 for i in range(self.cells_total)}
            self.parameters = self._init_cl_params(mode)
        # All cells in one cluster
        elif mode == 'together':
            self.assignment = np.zeros(self.cells_total, dtype=int)
            self.cells_per_cluster = {0: self.cells_total}
            self.parameters = self._init_cl_params(mode)
        # Complete random
        elif mode == 'random':
            self.assignment = np.random.randint(
                0, high=self.cells_total, size=self.cells_total
            )
            self.cells_per_cluster = {}
            cl, cl_size = np.unique(self.assignment, return_counts=True)
            for i in range(cl.size):
                bn.replace(self.assignment, cl[i], i)
                self.cells_per_cluster[i] = cl_size[i]
            self.parameters = self._init_cl_params(mode)
        else:
            raise TypeError(f'Unsupported Initialization: {mode}')
        self.init_DP_prior()


    def _init_cl_params(self, mode='random', fkt=1):
        params = np.zeros(self.data.shape)
        if mode == 'separate':
            params = np.random.beta(
                np.nan_to_num(self.p + self.data * fkt, \
                    nan=self._beta_mix_const[0]),
                np.nan_to_num(self.q + (1 - self.data) * fkt, \
                    nan=self._beta_mix_const[1])
            )
        elif mode == 'together':
            params[0] = np.random.beta(
                self.p + bn.nansum(self.data * fkt, axis=0),
                self.q + bn.nansum((1 - self.data) * fkt, axis=0)
            )
        elif mode == 'assign':
            for cl in self.cells_per_cluster:
                cl_data = self.data[np.where(self.assignment == cl)]
                params[cl] = np.random.beta(
                    self.p + bn.nansum(cl_data * fkt, axis=0),
                    self.q + bn.nansum((1 - cl_data)  * fkt, axis=0)
                )
        elif mode == 'random':
            k = np.unique(self.assignment)
            params[k] = np.random.uniform(size=(k.size, self.muts_total))
        return np.clip(params, TMIN, TMAX).astype(np.float32)


    def _init_cl_params_new(self, i, fkt=1):
        params = np.random.beta(
            self.p + bn.nansum(self.data[i] * fkt, axis=0),
            self.q + bn.nansum((1 - self.data[i]) * fkt, axis=0)
        )
        return np.clip(params, TMIN, TMAX).astype(np.float32)


    def init_DP_prior(self):
        cl_vals = np.append(np.arange(1, self.cells_total + 1), self.DP_a)
        CRP_prior = self.log_CRP_prior(cl_vals, self.cells_total, self.DP_a)
        self.CRP_prior = np.append(0, CRP_prior)


    def _calc_ll(self, x, theta, time=[-1], flat=False):
        ll_FN = theta * self._Bernoulli_FN(x, time)
        ll_FP = (1 - theta) * self._Bernoulli_FP(x, time)
        # 20240708 Liting: Add missing rate
        ll_Missing = self._Missing(x, time)
        #ll_full = np.log(ll_FN + ll_FP + ll_Missing)
        ll_full =  np.log(ll_FN + ll_FP + 10e-10)
        if flat:
            return bn.nansum(ll_full)
        else:
            return bn.nansum(ll_full, axis=1)


    #def _Bernoulli_FN(self, x):
    #    return (1 - self.FN) ** x * self.FN ** (1 - x)
    
    def _Bernoulli_FN(self, x, time = [-1]):
        # 20240627 Liting: calculate the FN for all time points
        FN = np.empty((x.shape[0], x.shape[1]))
        # for all cells
        if np.array_equal(np.array(time), [-1]):
            for i in range(len(self.FN)):
                idx = np.where(self.timepoint_x == i)[0]
                cl_x = x[idx, : ]
                #temp = (1 - self.FN[i] - self.Miss[i]) ** (cl_x) * self.FN[i] ** (1 - cl_x)
                temp = (1 - self.FN[i] ) ** (cl_x) * self.FN[i] ** (1 - cl_x)
                FN[idx, ] = temp
            return FN
        # for subset of cells
        for i in range(x.shape[0]):
            cl_x = x[i, :]
            #temp = (1 - self.FN[time[i]] - self.Miss[time[i]]) **  (cl_x) * self.FN[time[i]] ** (1 - cl_x)
            temp = (1 - self.FN[time[i]]) **  (cl_x) * self.FN[time[i]] ** (1 - cl_x)
            FN[i, ] = temp
        return FN

    #def _Bernoulli_FP(self, x):
    #    return (1 - self.FP) ** (1 - x) * self.FP ** x

    def _Bernoulli_FP(self, x, time=[-1]):
        # 20240627 Liting: calculate the FP for all time points
        # for all cell
        FP = np.empty((x.shape[0], x.shape[1]))
        if  np.array_equal(time, [-1]):
            for i in range(len(self.FP)):
                idx = np.where(self.timepoint_x == i)[0]
                cl_x = x[idx, : ]
                #temp = (1 - self.FP[i] - self.Miss[i]) ** (1 - cl_x) * self.FP[i] ** cl_x
                temp = (1 - self.FP[i] ) ** (1 - cl_x) * self.FP[i] ** cl_x
                FP[idx, ] = temp
            return FP
        # for subset of cells
        for i in range(x.shape[0]):
            cl_x = x[i, :]
            #temp = (1 - self.FP[time[i]] - self.Miss[time[i]]) ** (1 - cl_x) * self.FP[time[i]] ** cl_x
            temp = (1 - self.FP[time[i]]) ** (1 - cl_x) * self.FP[time[i]] ** cl_x
            FP[i, ] = temp
        return FP
    
    # 20240708 Liting: Add function to calculate missing rate
    def _Missing(self, x, time=[-1]):
        Miss = np.ones((x.shape[0], x.shape[1]))
        if np.array_equal(time, [-1]):
            for i in range(len(self.Miss)):
                idx = np.where(self.timepoint_x == i)[0]
                cl_x = x[idx, :]
                temp = np.where(np.isnan(cl_x), self.Miss[i], 1)
                Miss[idx, ] = temp
            return Miss
        for i in range(x.shape[0]):
            cl_x = x[i, :]
            temp = np.where(np.isnan(cl_x), self.Miss[time[i]], 1)
            Miss[i, ] = temp
        return Miss
    
    # 20240627 Liting: Commented out, not being used. 
    # def _Bernoulli_mut(self, x, theta):
    #     return x * (theta * (1 - self.FN) + (1 - theta) * self.FP)

    # 20240627 Liting: Commented out, not being used. 
    # def _Bernoulli_wt(self, x, theta):
    #     return (1 - x) * (theta * self.FN + (1 - theta) * (1 - self.FP))
 
    
    def get_lpost_single0(self, cell_id, cl_ids):
        # 20240627 Liting: Modified to take time as input
        time = self.timepoint_x[cell_id]
        ll_single = self._calc_ll(self.data[[cell_id]], self.parameters[cl_ids], [time])
        cl_size = np.fromiter(self.cells_per_cluster.values(), dtype=int)
        lprior = self.CRP_prior[cl_size]
        return ll_single + lprior
    
    def get_lpost_single(self, cell_id, cl_ids):
        time = self.timepoint_x[cell_id]
        ll_single = self._calc_ll(self.data[[cell_id]], self.parameters[cl_ids], [time])
        cl_size = np.fromiter(self.cells_per_cluster.values(), dtype=int)
        lprior = self.CRP_prior[cl_size]
        return ll_single + lprior
    

    '''
    # still give too many clusters
    def get_lpost_single(self, cell_id, cl_ids):
        ll_single = self.calc_ll_single(self.data[[cell_id]], self.parameters[cl_ids])
        cl_size = np.fromiter(self.cells_per_cluster.values(), dtype=int)
        lprior = self.CRP_prior[cl_size]
        return ll_single + lprior
    
    def calc_ll_single(self, x, theta, flat = False):
        fp = 0.01
        fn = 0.2
        FN = (1 - fn) ** x * fn ** (1 - x)
        FP = (1 - fp) ** (1 - x) * fp ** x
        ll_FP = (1 - theta) * FP
        ll_FN = theta * FN
        ll_full =  np.log(ll_FN + ll_FP )
        if flat:
            return bn.nansum(ll_full)
        else:
            return bn.nansum(ll_full, axis=1)
    '''

    '''   
    This approach make the result worse and takes much longer time 
    def get_cls_mean_error(self, time, cl_ids):
        mean_fp = []
        mean_fn = []
        curr_fp = self.FP[time]
        curr_fn = self.FN[time]
        for cl in cl_ids:
            cells = np.argwhere(self.assignment == cl).flatten()
            #print("cells", cells)
            #print("timepoints", self.timepoint_x[cells])
            fp = [self.FP[x] for x in self.timepoint_x[cells]]
            fn = [self.FN[x] for x in self.timepoint_x[cells]]
            fp.append(curr_fp)
            fn.append(curr_fn)
            mean_fp.append(np.median(fp))
            mean_fn.append(np.median(fn))
        return mean_fp, mean_fn
    
    def get_lpost_single1(self, cell_id, cl_ids):
        # 20240627 Liting: Modified to take time as input
        time = self.timepoint_x[cell_id]
        mean_fp, mean_fn = self.get_cls_mean_error(time, cl_ids)
        ll_single = self.calc_ll_single(self.data[[cell_id]], self.parameters[cl_ids], 
                                        mean_fp, mean_fn)
        cl_size = np.fromiter(self.cells_per_cluster.values(), dtype=int)
        lprior = self.CRP_prior[cl_size]
        return ll_single + lprior
    
    def calc_ll_single(self, x, theta, mean_fp, mean_fn, flat = False):
        FN = np.empty((len(mean_fn), x.shape[1]))
        FP = np.empty((len(mean_fp), x.shape[1]))
        for i in range(len(mean_fn)):
            FN[i, ] = (1 - mean_fn[i] ) ** (x) * mean_fn[i] ** (1 - x)
            FP[i, ] = (1 - mean_fp[i] ) ** (1 - x) * mean_fp[i] ** (x)
        ll_FP = (1 - theta) * FP
        ll_FN = theta * FN
        ll_full =  np.log(ll_FN + ll_FP )
        if flat:
            return bn.nansum(ll_full)
        else:
            return bn.nansum(ll_full, axis=1)

    '''
    def get_lpost_single_new_cluster(self):
        ll_FP = self._beta_mix_const[0] * self._Bernoulli_FP(self.data)
        ll_FN = self._beta_mix_const[1] * self._Bernoulli_FN(self.data)
        # 20240708 Liting: add missing rate
        #ll_Miss = self._Missing(self.data)
        ll_full = np.log(ll_FN + ll_FP)# + ll_Miss)
        return bn.nansum(ll_full, axis=1) + self.CRP_prior[-1]


    def get_ll_full(self):
        return self._calc_ll(self.data, self.parameters[self.assignment], flat = True)


    def get_lprior_full(self):
        lprior = self.DP_a_prior.logpdf(self.DP_a) \
            + bn.nansum(self.CRP_prior[
                np.fromiter(self.cells_per_cluster.values(), dtype=int)]
            )
        if not self.beta_prior_uniform:
            cl_ids = np.fromiter(self.cells_per_cluster.keys(), dtype=int)
            lprior += bn.nansum(
                self.param_prior.logpdf(self.parameters[cl_ids])
            )
        return lprior


    def update_assignments_Gibbs(self):
        """ Update the assignmen of cells to clusters by Gipps sampling

        """
        new_cl_post = self.get_lpost_single_new_cluster()
        test = np.zeros(self.cells_total)
        #print("update assignment with Gibbs============")
        counter = 0
        for cell_id in np.random.permutation(self.cells_total):
            # Remove cell from cluster
            old_cluster = self.assignment[cell_id]
            #print("cell", cell_id, "old cluster", old_cluster)
            #print(self.cells_per_cluster)
            if self.cells_per_cluster[old_cluster] == 1:
                del self.cells_per_cluster[old_cluster]
            else:
                self.cells_per_cluster[old_cluster] -= 1

            cl_ids = np.fromiter(self.cells_per_cluster.keys(), dtype=int)
            # Probability of joining an existing cluster
            post_old = self.get_lpost_single(cell_id, cl_ids)
            # Probability of starting a new cluster
            post_new = new_cl_post[cell_id]
            # Sample new cluster assignment from posterior
            sci = self.get_sci()
            #probs_norm = self._normalize_log_probs(np.append(post_old, post_new * (1 + sci)))
            probs_norm = self._normalize_log_probs(np.append(post_old, post_new * (1)))
            cl_ids = np.append(cl_ids, -1)
            new_cluster_id = np.random.choice(cl_ids, p=probs_norm)

            # Start a new cluster
            test[cell_id] = probs_norm[-1]
            if new_cluster_id == -1:
                # old cluster has more than one cell
                if old_cluster in self.cells_per_cluster:
                    new_cluster_id = self.init_new_cluster(cell_id)
                else: # old cluster has beend deleted
                    # self.parameters[old_cluster] keep unchange
                    # new_cluster_id = old_cluster
                    # if singleton, has to join some of the clusters
                    count = 0
                    while new_cluster_id == -1:
                        new_cluster_id = np.random.choice(cl_ids, p=probs_norm)  
                        count +=1
                        if count == 10: 
                            new_cluster_id = self.init_new_cluster(cell_id)
                            counter += 1
            # Assign to cluster
            self.assignment[cell_id] = new_cluster_id
            try:
                self.cells_per_cluster[new_cluster_id] += 1
            except KeyError:
                self.cells_per_cluster[new_cluster_id] = 1
            #print(self.cells_per_cluster)
            #print("cell", cell_id, "new cluster", self.assignment[cell_id])
        counter1 = np.fromiter(self.cells_per_cluster.keys(), dtype=int).shape[0]
        #print("start ", counter, "new cluster",  counter1)
    def init_new_cluster(self, cell_id):
        cl_id = self.get_empty_cluster()
        self.parameters[cl_id] = self._init_cl_params_new([cell_id])
        return cl_id


    def get_empty_cluster(self):
        return next(i for i in range(self.cells_total) 
            if i not in self.cells_per_cluster)


    def update_parameters(self, step_no=None):
        # Iterate over all populated clusters
        declined_t = np.zeros(len(self.cells_per_cluster), dtype= int)
        for i, cl_id in enumerate(self.cells_per_cluster):
            self.parameters[cl_id], _, declined = self.MH_cluster_params(
                self.parameters[cl_id],
                np.argwhere(self.assignment == cl_id).flatten()
            )
            declined_t[i] = declined
        return bn.nansum(declined_t), bn.nansum(self.muts_total - declined_t)


    def MH_cluster_params(self, old_params, cells, trans_prob=False):
        """ Update cluster parameters

        Arguments:
            old_parameter (float): old val of cluster parameter
            data (np.array): data for cells in the cluster

        Return:
            np.array: New cluster parameter
            float: Sum of MH decision paramters A
            int: Number of declined MH updates
        """

        # Propose new parameter from normal distribution
        std = np.random.choice(self.param_proposal_sd, size=self.muts_total)
        a = (TMIN - old_params) / std 
        b = (TMAX - old_params) / std
        new_params = truncnorm.rvs(a, b, loc=old_params, scale=std) \
            .astype(np.float32)

        A = self._get_log_A(new_params, old_params, cells, a, b, std, trans_prob)
        u = np.log(np.random.random(self.muts_total))

        decline = u >= A
        new_params[decline] = old_params[decline]

        if trans_prob:
            A[decline] = np.log(-1 * np.expm1(A[decline]))
            return new_params, bn.nansum(A), bn.nansum(decline)
        else:
            return new_params, np.nan, bn.nansum(decline)


    def _get_log_A(self, new_params, old_params, cells, a, b, std, clip=False):
        """ Calculate the MH acceptance paramter A
        """
        # Calculate the transition probabilitites
        new_p_target = truncnorm \
            .logpdf(new_params, a, b, loc=old_params, scale=std)

        a_rev = (TMIN - new_params) / std
        b_rev = (TMAX - new_params) / std
        old_p_target = truncnorm \
            .logpdf(old_params, a_rev, b_rev, loc=new_params, scale=std)

        # Calculate the log likelihoods
        ll_FN = self._Bernoulli_FN(self.data[cells], self.timepoint_x[cells])
        ll_FP = self._Bernoulli_FP(self.data[cells], self.timepoint_x[cells])
        # 20240708 Liting: Add missing rate
        #ll_miss = self._Missing(self.data[cells], self.timepoint_x[cells])
        '''
        new_ll = bn.nansum(
            np.log(new_params * ll_FN + (1 - new_params) * ll_FP + ll_miss), axis=0
        )
        old_ll = bn.nansum(
            np.log(old_params * ll_FN + (1 - old_params) * ll_FP + ll_miss), axis=0
        )
        '''
        new_ll = bn.nansum(
            np.log(new_params * ll_FN + (1 - new_params) * ll_FP ), axis=0
        )
        old_ll = bn.nansum(
            np.log(old_params * ll_FN + (1 - old_params) * ll_FP ), axis=0
        )
        # Calculate the priors
        if self.beta_prior_uniform:
            new_prior = 0
            old_prior = 0
        else:
            new_prior = self.param_prior.logpdf(new_params)
            old_prior = self.param_prior.logpdf(old_params)

        A = new_ll + new_prior - old_ll - old_prior + old_p_target - new_p_target

        if clip:
            return np.clip(A, a_min=None, a_max=0)
        else:
            return A


    def update_DP_alpha(self):
        """Escobar, D., West, M. (1995).
        Bayesian Density Estimation and Inference Using Mixtureq.
        Journal of the American Statistical Association, 90, 430.
        Chapter: 6. Learning about a and further illustration
        """
        k = len(self.cells_per_cluster)
        # Escobar, D., West, M. (1995) - Eq. 14
        eta = np.random.beta(self.DP_a + 1, self.cells_total)
        w = (self.DP_a_gamma[0] + k - 1) \
            / (self.cells_total * (self.DP_a_gamma[1] - np.log(eta)))
        pi_eta = w / (1 + w)

        # Escobar, D., West, M. (1995) - Eq. 13
        if np.random.random() < pi_eta:
            new_alpha = np.random.gamma(
                self.DP_a_gamma[0] + k, self.DP_a_gamma[1] - np.log(eta)
            )
        else:
            new_alpha = np.random.gamma(
                self.DP_a_gamma[0] + k - 1, self.DP_a_gamma[1] - np.log(eta)
            )

        self.DP_a = max(1 + EPSILON, new_alpha)
        #print("Update DP_a", self.DP_a)
        self.init_DP_prior()
        #print(self.CRP_prior[-1])


# ------------------------------------------------------------------------------
# SPLIT MERGE MOVE FOR NON CONJUGATES
# ------------------------------------------------------------------------------

    def update_assignments_split_merge(self, ratios=[.75, .25], step_no=5):
        """ Update the assignmen of cells to clusters by a split-merge move

        """
        cluster_no = len(self.cells_per_cluster)
        if cluster_no == 1:
            return (self.do_split_move(step_no), 0)
        elif cluster_no == self.cells_total:
            return (self.do_merge_move(step_no), 1)
        else:
            move = np.random.choice([0, 1], p=ratios)
            if move == 0:
                return (self.do_split_move(step_no), move)
            else:
                return (self.do_merge_move(step_no), move)


    def do_split_move(self, step_no=5):
        clusters = np.fromiter(self.cells_per_cluster.keys(), dtype=int)
        cluster_size = np.fromiter(self.cells_per_cluster.values(), dtype=int)
        # Chose larger clusters more often for split move
        cluster_probs = cluster_size / cluster_size.sum()

        # Get cluster with more than one item
        while True:
            clust_i = np.random.choice(clusters, p=cluster_probs)
            cells = np.argwhere(self.assignment == clust_i).flatten()
            if cells.size != 1:
                break

        # Get two random items from the cluster
        obs_i_idx, obs_j_idx = np.random.choice(cells.size, size=2, replace=False)
        cells[0], cells[obs_i_idx] = cells[obs_i_idx], cells[0]
        cells[-1], cells[obs_j_idx] = cells[obs_j_idx], cells[-1]

        # Eq. 3 in paper, second term
        cluster_idx = np.argwhere(clusters == clust_i).flatten()
        ltrans_prob_size = np.log(cluster_probs[cluster_idx]) \
            - np.log(self.cells_per_cluster[clust_i]) \
            - np.log(self.cells_per_cluster[clust_i] - 1)

        cluster_size_red = np.delete(cluster_size, cluster_idx)
        cluster_size_data = (ltrans_prob_size, cluster_size_red)

        accept, new_assignment, new_params = self.run_rg_nc(
            'split', cells, cluster_size_data, step_no
        )
        if accept:
            clust_new = self.get_empty_cluster()
            # Update parameters
            self.parameters[clust_i] = new_params[0]
            self.parameters[clust_new] = new_params[1]
            # Update assignment
            clust_new_cells = np.append(
                cells[1:-1][np.where(new_assignment == 1)], cells[-1]
            )
            self.assignment[clust_new_cells] = clust_new
            # Update cell-number per cluster
            self.cells_per_cluster[clust_i] -= clust_new_cells.size
            self.cells_per_cluster[clust_new] = clust_new_cells.size

            return [1, 0]
        else:
            return [0, 1]


    def do_merge_move(self, step_no=5):
        clusters = np.fromiter(self.cells_per_cluster.keys(), dtype=int)
        cluster_size = np.fromiter(self.cells_per_cluster.values(), dtype=int)
        # Chose smaller clusters more often for merge move
        cluster_size_inv = 1 / cluster_size
        cluster_probs = cluster_size_inv / cluster_size_inv.sum()
        cl_i, cl_j = np.random.choice(
            clusters, p=cluster_probs, size=2, replace=False
        )

        cells_i = np.argwhere(self.assignment == cl_i).flatten()
        obs_i_idx = np.random.choice(cells_i.size)
        cells_i[0], cells_i[obs_i_idx] = cells_i[obs_i_idx], cells_i[0]

        cells_j = np.argwhere(self.assignment == cl_j).flatten()
        obs_j_idx = np.random.choice(cells_j.size)
        cells_j[-1], cells_j[obs_j_idx] = cells_j[obs_j_idx], cells_j[-1]

        cells = np.concatenate((cells_i, cells_j)).flatten()

        # Eq. 6 in paper, second term
        ij_idx = np.argwhere((clusters == cl_j) | (clusters == cl_i)).flatten()
        cluster_size_data = bn.nansum(np.log(cluster_probs[ij_idx])) \
            - bn.nansum(np.log(cluster_size[ij_idx]))

        accept, new_params = self.run_rg_nc(
            'merge', cells, cluster_size_data, step_no
        )
        if len(cells_i) == 1 or len(cells_j) == 1:
            #print("Merging singleton clusters")
            accept = True
        if accept:
            # Update parameters
            self.parameters[cl_i] = new_params
            # Update Assignment
            self.assignment[cells_j] = cl_i
            # Update cells per cluster
            self.cells_per_cluster[cl_i] += cells_j.size
            del self.cells_per_cluster[cl_j]

            return [1, 0]
        else:
            return [0, 1]


    def run_rg_nc(self, move, cells, size_data, scan_no):   
        # Jain, S., Neal, R. (2007) - Section 4.2: 3,1,1
        self._rg_init_split(cells)
        # Jain, S., Neal, R. (2007) - Section 4.2: 3,2,1
        self.rg_params_merge = self._init_cl_params_new(cells)

        # Jain, S., Neal, R. (2007) - Section 4.2: 3,1,2 / 3,2,2
        # Do restricted Gibbs scans to reach y^{L_{split}} and y^{L_{merge}}
        for scan in range(scan_no):
            self._rg_scan_split(cells)
            self._rg_scan_merge(cells)

        # Jain, S., Neal, R. (2007) - Section 4.2: 4 or 5 (depending on move)
        # Do last scan to reach c_final and calculate Metropolis Hastings prob
        if move == 'split':
            #print("Doing split=============")
            return self._do_rg_split_MH(cells, size_data)
        else:
            #print("Doing merge=============")
            return self._do_rg_merge_MH(cells, size_data)


    def _rg_init_split(self, cells, random=False):
        i = cells[0]
        j = cells[-1]
        S = cells[1:-1]
        if S.size == 0:
            self.rg_assignment = np.array([])
        elif random:
            # assign cells to clusters i and j randomly
            self.rg_assignment = np.random.choice([0, 1], size=(S.size))
        else:
            # 20240628 Liting: added timepoint data to calculate ll
            ll_i = self._calc_ll(self.data[S],
                np.nan_to_num(self.data[i], nan=self._beta_mix_const[0]), self.timepoint_x[S])
            ll_j = self._calc_ll(self.data[S],
                np.nan_to_num(self.data[j], nan=self._beta_mix_const[0]), self.timepoint_x[S])
            self.rg_assignment = np.where(ll_j > ll_i, 1, 0)
        #initialize cluster parameters
        cells_i = np.append(S[np.argwhere(self.rg_assignment == 0)], i)
        cells_j = np.append(S[np.argwhere(self.rg_assignment == 1)], j)
        cl_i_params = self._init_cl_params_new(cells_i)
        cl_j_params = self._init_cl_params_new(cells_j)
        self.rg_params_split = np.stack([cl_i_params, cl_j_params])


    def _rg_scan_split(self, cells, trans_prob=False):
        if cells.size == 2:
            prob_cl = 0 #log_EPSILON
        else:
            prob_cl = self._rg_scan_assign(cells, trans_prob)
        prob_par = self._rg_scan_params(cells, trans_prob)
        
        if trans_prob:
            return prob_cl + prob_par


    def _rg_scan_merge(self, cells, trans_prob=False):
        # Update cluster parameters
        self.rg_params_merge, prob, _ = self.MH_cluster_params(
            self.rg_params_merge, cells, trans_prob
        )
        if trans_prob:
            return prob


    def _rg_scan_params(self, cells, trans_prob=False):
        # Update parameters of cluster i and j
        i = cells[0]
        j = cells[-1]
        S = cells[1:-1]
        prob = np.zeros(2)
        for cl in range(2):
            if cl == 0:
                cl_cells = np.append(S[np.argwhere(self.rg_assignment == 0)], i)
            else:
                cl_cells = np.append(S[np.argwhere(self.rg_assignment == 1)], j)
            self.rg_params_split[cl], prob[cl], _ = self.MH_cluster_params(
                self.rg_params_split[cl], cl_cells, trans_prob
            )

        if trans_prob:
            return prob.sum()


    def _rg_scan_assign(self, cells, trans_prob=False):
        ll = self._rg_get_ll(cells[1:-1], self.rg_params_split)
        n = cells.size
        if trans_prob:
            prob = np.zeros(n - 2)
        
        # Iterate over all obersavtions k
        for cell in np.random.permutation(n - 2):
            self.rg_assignment[cell] = -1
            # Get normalized log probs of assigning an obs. to clusters i or j
            # +1 to compensate obs = -1; +1 for observation j
            n_j = bn.nansum(self.rg_assignment) + 2
            n_i = n - n_j - 1
            log_post = ll[cell] + self.log_CRP_prior([n_i, n_j], n, self.DP_a)
            log_probs = self._normalize_log(log_post)
            # Sample new cluster assignment from posterior
            new_clust = np.random.choice([0, 1], p=np.exp(log_probs))
            
            self.rg_assignment[cell] = new_clust
            if trans_prob:
                prob[cell] = log_probs[new_clust]

        if trans_prob:
            return bn.nansum(prob)


    def _rg_get_ll(self, cells, params):
        # 20240628 Liting: Add timepoint. Cells are idx here 
        ll_i = self._calc_ll(self.data[cells], params[0], self.timepoint_x[cells])
        ll_j = self._calc_ll(self.data[cells], params[1], self.timepoint_x[cells])
        return np.stack([ll_i, ll_j], axis=1)


    def _do_rg_split_MH(self, cells, size_data):
        A = self._get_trans_prob_ratio_split(cells) \
            + self._get_lprior_ratio_split(cells) \
            + self._get_ll_ratio(cells, 'split') \
            + self._get_ltrans_prob_size_ratio_split(*size_data)

        if np.unique(self.rg_assignment).size == 1:
            return (False, [], [])
        
        # 20240718 Liting: don't accept if result in singleton clusters
        unique_elements, counts = np.unique(self.rg_assignment, return_counts=True)
        if np.any(counts == 1):
            #print("Resulting singleton clusters")
            return (False, [], [])
        if np.log(np.random.random()) < A:
            #print("Accept split", A)
            return (True, self.rg_assignment, self.rg_params_split)
        #else:
            #print("Reject split", A)
        return (False, [], [])


    def _do_rg_merge_MH(self, cells, size_data):
        A = self._get_trans_prob_ratio_merge(cells) \
            + self._get_lprior_ratio_merge(cells) \
            + self._get_ll_ratio(cells, 'merge') \
            + self._get_ltrans_prob_size_ratio_merge(size_data)

        if np.log(np.random.random()) < A:
            #print("Accept merge", A)
            return (True, self.rg_params_merge)
        #else:
            #print("Reject merge", A)
        return (False, self.rg_params_merge)


    def _get_trans_prob_ratio_split(self, cells):
        """ [eq. 15 in Jain and Neal, 2007]
        """
        # Do split GS: Launch to proposal state
        GS_split = self._rg_scan_split(cells, trans_prob=True)
        # Do merge GS: Launch to original state
        std = np.random.choice(self.param_proposal_sd, size=self.muts_total)
        a = (TMIN - self.rg_params_merge) / std
        b = (TMAX - self.rg_params_merge) / std

        GS_merge = bn.nansum(
            self._get_log_A(self.parameters[self.assignment[cells[0]]],
                self.rg_params_merge, cells, a, b, std, True)
        )
        return GS_merge - GS_split


    def _get_trans_prob_ratio_merge(self, cells):
        """ [eq. 16 in Jain and Neal, 2007]
        """
        # Do merge GS
        GS_merge = self._rg_scan_merge(cells, trans_prob=True)
        # Do split GS to original merge state
        GS_split = self._rg_get_split_prob(cells)
        return GS_split - GS_merge
      

    def _get_lprior_ratio_split(self, cells):
        """ [eq. 7 in Jain and Neal, 2007]
        """
        n = self.rg_assignment.size + 2
        n_j = bn.nansum(self.rg_assignment) + 1
        n_i = n - n_j
        # Cluster assignment prior
        lprior_rate = np.log(self.DP_a) - gammaln(n)
        if n_i > 0:
            lprior_rate += gammaln(n_j)
        if n_j > 0:
            lprior_rate += gammaln(n_i)
        # Cluster parameter prior
        if not self.beta_prior_uniform:
            cl_id = self.assignment[cells[0]]
            lprior_rate += \
                    bn.nansum(self.param_prior.logpdf(self.rg_params_split)) \
                - bn.nansum(self.param_prior.logpdf(self.parameters[cl_id]))
        return lprior_rate


    def _get_ll_ratio(self, cells, move):
        """ [eq. 11/eq. 12 in Jain and Neal, 2007]
        """
        i_ids = np.append(
            cells[1:-1][np.argwhere(self.rg_assignment == 0)], cells[0]
        )
        j_ids = np.append(
            cells[1:-1][np.nonzero(self.rg_assignment)], cells[-1]
        )

        ll_i = self._calc_ll(self.data[i_ids], self.rg_params_split[0], self.timepoint_x[i_ids], True)
        ll_j = self._calc_ll(self.data[j_ids], self.rg_params_split[1], self.timepoint_x[j_ids], True)
        ll_all = self._calc_ll(self.data[cells], self.rg_params_merge, self.timepoint_x[cells], True)

        if move == 'split':
            return ll_i + ll_j - ll_all
        else:
            return ll_all - ll_i - ll_j


    def _get_lprior_ratio_merge(self, cells):
        """ [eq. 8 in Jain and Neal, 2007]
        """
        n = cells.size
        n_j = bn.nansum(self.rg_assignment) + 1
        n_i = n - n_j
        # Cluster priors
        lprior_rate = gammaln(n) - np.log(self.DP_a)
        if n_i > 0:
            lprior_rate -= gammaln(n_i)
        if n_j > 0:
            lprior_rate -= gammaln(n_j)
        # Parameter priors
        if not self.beta_prior_uniform:
            cl_ids = self.assignment[[cells[0], cells[-1]]]
            lprior_rate += \
                    bn.nansum(self.param_prior.logpdf(self.rg_params_merge)) \
                - bn.nansum(self.param_prior.logpdf(self.parameters[cl_ids]))
        return lprior_rate


    def _get_ltrans_prob_size_ratio_split(self, ltrans_prob_size, cluster_size):
        n_j = bn.nansum(self.rg_assignment) + 1
        n_i = self.rg_assignment.size + 2 - n_j 

        # Eq. 5 paper, first term
        norm = bn.nansum(1 / np.append(cluster_size, [n_i, n_j]))
        ltrans_prob_rev = np.log(1 / n_i / norm) + np.log(1 / n_j / norm)
        return ltrans_prob_rev - ltrans_prob_size[0] 


    def _get_ltrans_prob_size_ratio_merge(self, trans_prob_size):
        # Eq. 6, paper
        try:
            ltrans_prob_rev = -np.log(self.cells_total) \
                - np.log(self.rg_assignment.size - 1)
        except FloatingPointError:
            ltrans_prob_rev = -np.log(self.cells_total)
        return ltrans_prob_rev - trans_prob_size


    def _rg_get_split_prob(self, cells):
        std = np.random.choice(self.param_proposal_sd, size=(2, self.muts_total))
        a = (0 - self.rg_params_split) / std
        b = (1 - self.rg_params_split) / std

        i = cells[0]
        cl_i = self.assignment[i]
        j = cells[-1]
        cl_j = self.assignment[j]
        S = cells[1:-1]
        # Get paramter transition probabilities
        prob_param_i = bn.nansum(self._get_log_A(
            self.parameters[cl_i], self.rg_params_split[0],
            np.append(S[np.argwhere(self.rg_assignment == 0)], i),
            a[0], b[0], std[0], True
        ))
        prob_param_j = bn.nansum(self._get_log_A(
            self.parameters[cl_j], self.rg_params_split[1],
            np.append(S[np.argwhere(self.rg_assignment == 1)], j),
            a[1], b[1], std[1], True
        ))

        # Get assignment transition probabilities
        ll = self._rg_get_ll(
            cells[1:-1], (self.parameters[cl_i], self.parameters[cl_j])
        )
        n = cells.size
        prob_assign = np.zeros(S.size)

        assign = np.where(self.assignment[S] == cl_i, 0, 1)
        # Iterate over all obersavtions k != [i,j]
        for obs in range(S.size):
            self.rg_assignment[obs] = -1
            n_j = bn.nansum(self.rg_assignment) + 2
            n_i = n - n_j - 1

            # Get normalized log probs of assigning an obs. to clusters i or j
            log_post = ll[obs] + self.log_CRP_prior([n_i, n_j], n, self.DP_a)
            log_probs = self._normalize_log(log_post)
            # assign to original cluster and add probability
            self.rg_assignment[obs] = assign[obs]
            prob_assign[obs] = log_probs[assign[obs]]

        return prob_param_i + prob_param_j + bn.nansum(prob_assign)


if __name__ == '__main__':
    print('Here be dragons....')
