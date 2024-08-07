# BnpC
Bayesian non-parametric clustering (BnpC) of binary data with missing values and uneven error rates.

BnpC is a novel non-parametric method to cluster individual cells into clones and infer their genotypes based on their noisy mutation profiles.
BnpC employs a Chinese Restaurant Process prior to handle the unknown number of clonal populations. The model introduces a combination of Gibbs sampling, a modified non-conjugate split-merge move and Metropolis-Hastings updates to explore the joint posterior space of all parameters. Furthermore, it employs a novel estimator, which accounts for the shape of the posterior distribution, to predict the clones and genotypes.

The corresponsing paper can be found in [Bioinformatics](https://doi.org/10.1093/bioinformatics/btaa599 "Borgsmueller et al.")

# Contents
- [Installation](#Installation)
- [Usage](#Usage)
- [Example data](#Example-data)

# Requirements
- Python 3.X

# Installation
## Clone repository
First, download BnpC from github and change to the directory:
```bash
git clone https://github.com/cbg-ethz/BnpC
cd BnpC
```

## Create conda environment (optional)
First, create a new environment named "BnpC":
```bash
conda create --name BnpC python=3
```

Second, source it:
```bash
conda activate BnpC
```

## Install requirements
Use pip to install the requirements:
```bash
python -m pip install -r requirements.txt
```

Now you are ready to run **BnpC**!

# Usage
The BnpC wrapper script `run_BnpC.py` can be run with the following shell command:
```bash
python run_BnpC.py <INPUT_DATA> [-t] [-FN] [-FP] [-FN_m] [-FN_sd] [-FP_m] [-FP_sd] [-dpa] [-pp] [-n] [-s] [-r] [-ls] [-b] [-smp] [-cup] [-e] [-sc] [--seed] [-o] [-v] [-np] [-tr] [-tc] [-td]]
```

## Input
BnpC requires a binary matrix as input, where each row corresponds with a mutations and each columns with a cell.
All matrix entries must be of the following: 0|1|3/" ", where 0 indicates the absence of a mutation, 1 the presence, and a 3 or empty element a missing value.

> ## Note
> If your data is arranged in the transposed way (cells = columns, rows = mutations), use the `-t` argument.

## Arguments
### Input Data Arguments
- `<str>`, Path to the input data.
- `-t <flag>`, If set, the input matrix is transposed.

### Optional input arguments (for simulated data)
- `-tr <str>`, Path to the mutation tree file (in .gv format) used for data generation.
- `-tc <str>`, Path to the true clusters assignments to compare clustering methods.
- `-td <str>`, Path to the true/raw data/genotypes.

### Model Arguments
- `-FN <float>`, Replace <float\> with the fixed error rate for false negatives.
- `-FP <float>`, Replace <float\> with the fixed error rate for false positives.
- `-FN_m <float>`, Replace <float\> with the mean for the prior for the false negative rate.
- `-FN_sd <float>`, Replace <float\> with the standard deviation for the prior for the false negative rate.
- `-FP_m <float>`, Replace <float\> with the mean for the prior for the false positive rate.
- `-FP_sd <float>`, Replace <float\> with the standard deviation for the prior for the false positive rate.
- `-ap <float>`, Alpha value of the Beta function used as prior for the concentration parameter of the CRP.
- `-pp <float> <float>`, Beta function shape parameters used for the cluster parameter prior.

> ## Note
> If you run BnpC on panel data with **few mutation** only or on **error free** data, we recommend changing the `-pp` argument to beta distribution closer to uniform, like `-pp 0.75 0.75` or even `-pp 1 1`. Otherwise, BnpC will incorrectly report many singleton clusters.

### MCMC Arguments
- `-n <int>`, Number of MCMC chains to run in parallel (1 chain per thread).
- `-s <int>`, Number of MCMC steps.
- `-r <int>`, Runtime in minutes. If set, steps argument is overwritten.
- `-ls <float>`, Lugsail batch means estimator as convergence diagnostics [Vats and Flegal, 2018].
- `-b  <float>`, Ratio of MCMC steps discarded as burn-in.
- `-cup  <float>`, Probability of updating the CRP concentration parameter.
- `-eup <float>`, Probability to do update the error rates in An MCMC step.
- `-smp <float>`, Probability to do a split/merge step instead of Gibbs sampling.
- `-sms <int>`, Number of intermediate, restricted Gibbs steps in the split-merge move.
- `-smr <float, float>`, Ratio of splits/merges in the split merge move.
- `-e +<str>`, Estimator(s) for inferrence. If more than one, seperate by space. Options = posterior|ML|MAP.
- `-sc <flag>`, If set, infer a result for each chain individually (instead of from all chains together).
- `--seed <int>`, Seed used for random number generation.

### Output Arguments
- `-o <str>`, Path to an output directory.
- `-np <flag>`, If set, no plots are generated.
- `-v <int>`, Stdout verbosity level. Options = 0|1|2.

# Example data

Lets employ the toy dataset that one can find in the `data` folder (data.csv) to understand the functionality of the different arguments. First go to the folder and activate the environment:

        cd /path/to/crp_clustering
        conda activate environment_name

BnpC can run in three different settings:
1. Number of steps. Runs for the given number of MCMC steps. Arument: -s
2. Running time limit. Every MCMC the time is tracked and the method stops after the introduced time is achieved. Argument: -r
3. Lugsail for convergence diagnosis. The chain is terminated if the estimator undercuts a threshold defined by a significance level of 0.05 and a user defined float between [0,1], comparable to the half-width of the confidence interval in sample size calculation for a one sample t-test. Reasonal values = 0.1, 0.2, 0.3. Argument: -ls

The simplest way to run the BnpC is to leave every argument as default and hence only the path to the data needs to be given. In this case BnpC runs in the setting 1.
```bash
python run_BnpC.py example_data/data.csv 
```
If the error rates are known for a particular sequenced data (e.g FP = 0.0001 and FN = 0.3), one can run BnpC with fixed error rates by:
```bash
python run_BnpC.py example_data/data.csv -FP 0.0001 -FN 0.3
```
On the other hand, if errors are not known one can leave it blank as in the first case or if there is some intuition add the mean and standard deviation priors for the method to learn them:
```bash
python run_BnpC.py example_data/data.csv -FP_m 0.0001 -FN_m 0.3 -FP_sd 0.000001 -FN_sd 0.05
```
Additional MCMC arguments can be employed to allow faster convergence. Among other options:
- Reduce burnin to include more posterior samples in the estimation. Example: -b 0.2, discard 20 % of the total MCMC steps.
- Adapt split-merge probability to better explore the posterior landscape. Example: -smp 0.33, 1 out of every 3 steps will be a split-merge move on average.
- Adjust the Dirchlet Process alpha which accounts for the probability of starting a new cluster. Example: -dpa 10. Increasing the value, leads to a larger probability of starting a new cluster in the cell assignment step.

# Below is the details of modified changes.

### Example data
- large_t.csv file is the final inout file which I used to execute the program. It is a tab seperated file with timepoint data.

### CRP_learning_errors.py
- Converted alpha and beta variable into an array. Previously initialized as a single value, conversion implies it's intended to be an array.

### CRP.py
- Added a comment indicating the modification for beta prior check and DP alpha check.
- The Bernoulli functions '_Bernoulli_FN' and '_Bernoulli_FP' are modified to accept an additional parameter 'timepoint_x'. These functions now consider different false negative and false positive rates for each timepoint, accessed through the arrays using the 'timepoint_x' parameter.
- 'get_log_A' function is modified to accept a 'timepoint_x' parameter and the Bernoulli functions are called with the 'timepoint_x' parameter, allowing them to compute probabilities based on the FN and FP rates specific to the given timepoint.

### MCMC.py
- Added 'try-except' block to handle the situations where 'self.burn_in' might not be  defined or is of inappropriate type. (Program was being terminated here. Hence, I added this block. Code might need more modifications in future)

### dpmmIO.py
- Modified the load_data method to consider the new input file with timepoint data. Here, I extracted the values from the last column of the DataFrame and stores them in the 'timepoint_x' variable. Finally, the variable is returned along with the data values.
- Retreived the time point information by calling the updated load_data function.

### run_BnpC.py
- 'main' function is modified to unoack the rerun values of 'load_data' function. With the modification, the return values of 'load_data' are unpacked into 'data', 'timepoint_x', 'row_names', and 'col_names'.
- The 'generate_output' function is now called with modified argument list. This change ensures that the function receives all necessary information to generate the output.




# Changes by Liting
## Things to do
- Change 'load_data' to read timepoint properly
- Change FP, FN to arrays in CRP class
- Add Missing arrays to CRP class
- Change error rate learning process
- Modify probability calculation

### Changes: 20240625
- Changed the 'load_data' to extract the time point before replacing '3' with nan as '3' could be a timepoint in the last column. This avoid 'nan' in time point data.
- Added a new argument 'times' to know how many times points are there in the data.
- Added a new arguemnt 'Miss' for missing rate.
- Changed the argument of FP, FN, and Miss to take a list of number for each time point.

### Changes: 20240626
- Modified 'MH_error_rates()' in 'CRP_learning_errors.py' to include lists for error rates for different times points and missing rate. 
- Modified 'get_ll_full_error()' in 'CRP_learning_errors.py' to calculate the likelihood for each time point and include missing rate. 
