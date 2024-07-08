#!/usr/bin/env python3

import numpy as np
from scipy.stats import beta, truncnorm
import bottleneck as bn
import copy

try:
    from libs.CRP import CRP
except ImportError:
    from CRP import CRP


# ------------------------------------------------------------------------------
# LEARNING ERROR RATES - NO NANs
# ------------------------------------------------------------------------------

class CRP_errors_learning(CRP):
    #Bhavya Changes here // Updated DP_alpha to an array
    def __init__(self, data, timepoint_x, DP_alpha=[1], param_beta=[1, 1], \
                FP_mean=0.001, FP_sd=0.0005, FN_mean=0.25, FN_sd=0.05, \
                Miss_mean = 0.25, Miss_sd = 0.05, num_times = 1):
        super().__init__(data, timepoint_x, DP_alpha, param_beta, [FN_mean]*num_times, [FP_mean]*num_times, \
                        [Miss_mean]*num_times, num_times=num_times)
        # Error rate prior
        FP_trunc_a = (0 - FP_mean) / FP_sd
        FP_trunc_b = (1 - FP_mean) / FP_sd
        self.FP_prior = truncnorm(FP_trunc_a, FP_trunc_b, FP_mean, FP_sd)
        # self.FP_prior = beta(1, (1 - FP_mean) / FP_mean)
        self.FP_sd = np.array([FP_sd * 0.5, FP_sd, FP_sd * 1.5])

        FN_trunc_a = (0 - FN_mean) / FN_sd
        FN_trunc_b = (1 - FN_mean) / FN_sd
        self.FN_prior = truncnorm(FN_trunc_a, FN_trunc_b, FN_mean, FN_sd)
        # self.FN_prior = beta(1, (1 - FN_mean) / FN_mean)
        self.FN_sd = np.array([FN_sd * 0.5, FN_sd, FN_sd * 1.5])

        # 20240626 Liting: add missing rate
        Miss_trunc_a = (0 - Miss_mean) / Miss_sd
        Miss_trunc_b = (1 - Miss_mean) / Miss_sd
        self.Miss_prior = truncnorm(Miss_trunc_a, Miss_trunc_b, Miss_mean, Miss_sd)
        self.Miss_sd = np.array([Miss_sd * 0.5, Miss_sd, Miss_sd * 1.5])

    def __str__(self):
        out_str = '\nDPMM with:\n' \
            f'\t{self.cells_total} cells\n\t{self.muts_total} mutations\n' \
            f'\tlearning errors\n' \
            '\n\tPriors:\n' \
            f'\tGround Truth params.:\tBeta({self.p},{self.q})\n' \
            f'\tCRP a_0:\tGamma({self.DP_a_gamma[0]:.2f},{self.DP_a_gamma[1]})\n' \
            f'\tFP:\t\ttrunc norm({self.FP_prior.args[2]},{self.FP_prior.args[3]})\n' \
            f'\tFN:\t\ttrunc norm({self.FN_prior.args[2]},{self.FN_prior.args[3]})\n'\
            f'\tMiss:\t\ttrunc nurm({self.Miss_prior.args[2]},{self.Miss_prior.args[3]})\n'
        return out_str

    '''
    def get_lprior_full(self):
        return super().get_lprior_full() \
            + self.FP_prior.logpdf(self.FP) + self.FN_prior.logpdf(self.FN) +  self.Miss_prior.logpdf(self.Miss)
    '''
# NEED: think about how to calcualte the prior probability given multiple FP and FN?
# multiply? or addition? or mean?
    def get_lprior_full(self):
        return super().get_lprior_full() \
            + np.mean(self.FP_prior.logpdf(self.FP)) + np.mean(self.FN_prior.logpdf(self.FN)) \
               +  np.mean(self.Miss_prior.logpdf(self.Miss))

    def update_error_rates(self):
        self.FP, FP_count = self.MH_error_rates('FP')
        self.FN, FN_count = self.MH_error_rates('FN')
        # 20240626 Liting: add missing rate
        self.Miss, Miss_count = self.MH_error_rates('Miss')
        return FP_count, FN_count, Miss_count


    def get_ll_full_error(self, FP, FN, Miss):
        par = self.parameters[self.assignment]
        # 20240626 Liting: commented out to include missing rate
        ''' 
        without missing 
        (1 - FN) ** self.data # true positive 
        FN ** (1 - self.data) # false negative 
        (1 - FP) ** (1 - self.data) # true negative
        FP ** self.data # false positive
        ll_FN = par * (1 - FN) ** self.data * FN ** (1 - self.data)
        ll_FP = (1 - par) * (1 - FP) ** (1 - self.data) * FP ** self.data
        with missing rate
        (1 - FN - Miss) ** self.data # true positive 
        FN ** (1 - self.data) # false negative 
        (1 - FP - Miss) ** (1 - self.data) # true negative
        FP ** self.data # false positive
        ll_FN = par * (1 - FN) ** self.data * FN ** (1 - self.data)
        ll_FP = (1 - par) * (1 - FP) ** (1 - self.data) * FP ** self.data
        '''
        # Adjusted for missing rate
        # NEED: Think about how to calculate the likelihood of missing rate
        ll_full = 0
        ll_FN = np.empty((self.data.shape[0], self.data.shape[1]))
        ll_FP = np.empty((self.data.shape[0], self.data.shape[1]))
        ll_Miss = np.empty((self.data.shape[0], self.data.shape[1]))
        for i in range(len(FP)):
            # find index of cells in current time point
            idx = np.where(self.timepoint_x == i)[0]
            cl_data = self.data[idx, :]
            cl_par = par[idx, :]
            ll_FN[idx, ]=   (cl_par * (1 - FN[i] - Miss[i]) ** cl_data * FN[i] ** (1 - cl_data))
            ll_FP[idx, ] =  ((1 - cl_par) * (1 - FP[i] - Miss[i]) ** (1 - cl_data) * FP[i] ** cl_data)
            ll_Miss[idx, ] = np.where(np.isnan(cl_data), Miss[i], 1)
        
        ll_full = np.log(ll_FN + ll_FP + ll_Miss)
        #ll_full = np.log(ll_FN + ll_FP)
        return bn.nansum(ll_full)


    def MH_error_rates(self, error_type):
        # Set error specific values
        if error_type == 'FP':
            old_errors = self.FP
            prior = self.FP_prior
            stdevs = self.FP_sd
        # 20240626 Liting: Adjusted for missing rate
        elif error_type == "FN":
            old_errors = self.FN
            prior = self.FN_prior
            stdevs = self.FN_sd
        else:
            old_errors = self.Miss
            prior = self.Miss_prior
            stdevs = self.Miss_sd

        # Get new error from proposal distribution
        # 20240626 Liting: Sample one timepoint to update
        i = np.random.randint(0, self.num_times)
        old_error = old_errors[i]
        std = np.random.choice(stdevs)
        a = (0 - old_error) / std
        b = (1 - old_error) / std
        try:
            new_error = truncnorm.rvs(a, b, loc=old_error, scale=std)
        except FloatingPointError:
            new_error = truncnorm.rvs(a, np.inf, loc=old_error, scale=std)

        # Calculate transition probabilitites
        new_p_target = truncnorm \
            .logpdf(new_error, a, b, loc=old_error, scale=std)
        a_rev, b_rev = (0 - new_error) / std, (1 - new_error) / std
        old_p_target = truncnorm \
            .logpdf(old_error, a_rev, b_rev, loc=new_error, scale=std)
        
        # 20240626 Liting: get new error list
        new_errors = copy.deepcopy(old_errors)
        new_errors[i] = new_error
        # Calculate likelihood
        # 20240626 Liting: Added missing rate 
        if error_type == 'FP':
            new_ll = self.get_ll_full_error(new_errors, self.FN, self.Miss)
            old_ll = self.get_ll_full_error(old_errors, self.FN, self.Miss)
        elif error_type == "FN":
            new_ll = self.get_ll_full_error(self.FP, new_errors, self.Miss)
            old_ll = self.get_ll_full_error(self.FP, old_errors, self.Miss)
        else:
            new_ll = self.get_ll_full_error(self.FP, self.FN, new_errors)
            old_ll = self.get_ll_full_error(self.FP, self.FN, old_errors)
        # Calculate priors
        new_prior = prior.logpdf(new_error)
        old_prior = prior.logpdf(old_error)

        # Calculate MH decision treshold
        A = new_ll + new_prior - old_ll - old_prior + old_p_target - new_p_target

        if np.log(np.random.random()) < A:
            return new_errors, [1, 0]

        return old_errors, [0, 1]


if __name__ == '__main__':
    print('Here be dragons...')