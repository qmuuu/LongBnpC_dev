import os
from sklearn.metrics.cluster import v_measure_score
import numpy as np
import sys
from copy import deepcopy
import pandas as pd 

exps = ["fn_0.1", "fn_0.3", "fnSD_0.05", "fnSD_0.15", "fp_0.001", "fp_0.05",
"fpSD_0.01", "fpSD_0.03", "mc_3.5", "mc_5.1", "miss_0.1", "miss_0.3",
"missSD_0.05", "missSD_0.15", "mu_0.1", "mu_0.3", "time_3", "time_4", "time_5"]

header = ['rep', 'fn', 'fp', 'fnSD', 'fpSD', 'mc', 'miss',
          'missSD', 'mu', 'time', 'method']

BnpC_ext = "/gpfs/research/fangroup/lz20w/BnpC_ext/BnpC_ext_results/"
BnpC = "/gpfs/research/fangroup/lz20w/BnpC_ext/BnpC_results/"
BnpC1 = "/gpfs/research/fangroup/lz20w/BnpC_ext/BnpC1_results/"
true_dir = "/gpfs/research/fangroup/lz20w/BnpC_ext/sim_data/"

default = {
  'rep': '',
  'fp': '0.01',
  'fn': '0.2',
  'fpSD': '0.02',
  'fnSD': "0.1",
  'mc': "1.7",
  'miss': '0.2',
  'missSD': "0.1",
  'mu': "0.2",
  'time': "3",
  'method': '',
  'V_measure': '0'}

df = None
for exp in exps:
  for j in range(1, 6):
    rep = "rep" + str(j)
    path = exp + "/" + rep
    trueF = true_dir + path + "/true_cluster.tsv"
    true = []
    print(trueF)
    if os.path.exists(trueF):
      f = open(trueF, "r")
      assignment = {}
      total = 0
      for line in f.readlines():
        label = int(line.rstrip().split()[0])
        cells = [int(x) for x in line.rstrip().split()[1].split(";")]
        assignment[label] = cells
        total += len(cells)
      f.close()
      true = np.array([-1] * total)
      for key, cells in assignment.items():
        true[cells] = key
    temp = deepcopy(default)
    val = exp.split("_")[1]
    temp[exp.split("_")[0]] = val
    temp['rep'] = rep
    if exp.split("_")[0] == 'time':
      temp['time'] = int(exp.split("_")[1]) - 1

    BnpC_result = BnpC + path + "/assignment.txt"
    print(BnpC_result)
    if os.path.exists(BnpC_result):
      f = open(BnpC_result, "r")
      line = f.readlines()[1].rstrip().split()
      pred = [int(x) for x in line[2:]]
      v_measure = v_measure_score(true, pred)
      temp['method'] = 'BnpC'
      temp['V_measure'] = v_measure
      if df is None:
        df = pd.DataFrame(columns=default.keys())
        df = df.append(temp, ignore_index=True)
      else:
        df = df.append(temp, ignore_index = True)
    
    BnpC1_result = BnpC1 + path + "/assignment.txt"
    print(BnpC1_result)
    if os.path.exists(BnpC1_result):
      f = open(BnpC1_result, "r")
      line = f.readlines()[1].rstrip().split()
      pred = [int(x) for x in line[2:]]
      v_measure = v_measure_score(true, pred)
      temp['method'] = 'BnpC1'
      temp['V_measure'] = v_measure
      if df is None:
        df = pd.DataFrame(columns=default.keys())
        df = df.append(temp, ignore_index=True)
      else:
        df = df.append(temp, ignore_index = True)
    BnpC_ext_result =  BnpC_ext + path + "/assignment.txt"
    print(BnpC_ext_result)
    if os.path.exists(BnpC_ext_result):
      f = open(BnpC_ext_result, "r")
      line = f.readlines()[1].rstrip().split()
      pred = [int(x) for x in line[2:]]
      v_measure = v_measure_score(true, pred)
      temp['method'] = 'BnpC_ext'
      temp['V_measure'] = v_measure
      if df is None:
        df = pd.DataFrame(columns=default.keys())
        df = df.append(temp, ignore_index=True)
      else:
        df = df.append(temp, ignore_index = True)
df.to_csv("merged_results.tsv", index = False, sep = "\t")
