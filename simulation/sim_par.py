#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@authors: Xian Fan Mallory
Contacting email: fan@cs.fsu.edu
"""

import sys
import argparse
import numpy as np
import random
import copy
from collections import OrderedDict
from scipy.stats import truncnorm

class Edge:
    def __init__(self, p, c):
        self.p = p
        self.c = c

class Node:
    def __init__(self, e, p, c):
        self.e = e
        self.p = p
        self.c = c

def init_matrix(n, m):
    return [[0] * m for _ in range(n)]

def save_matrix(matrix, matrix_file):
    with open(matrix_file, "w") as f:
        for row in matrix:
            f.write("\t".join(map(str, row)) + "\n")

def get_mutation_count(lambda_value):
    mut_count = 0
    while mut_count == 0:
        mut_count = np.random.poisson(lam=lambda_value)
    return mut_count

def save_dict_to_file(data_dict, out_file):
    with open(out_file, 'w') as f:
        for key, value in data_dict.items():
            f.write(f"{key}\t{value}\n")

def distribute_mutations(tree_file, mutation_constant, out_file):
    tree_dict = {}
    with open(tree_file, "r") as f:
        f.readline()  # Skip header
        for line in f:
            line = line.strip()
            if line:
                fields = line.split('\t')
                nodeID = int(fields[1])
                tree_dict[nodeID] = f"{fields[2]};{fields[3]};{fields[6]}"
    
    mut_dict = {}
    prev_mut = -1
    for nodeID, value in tree_dict.items():
        if nodeID == 0:
            continue
        parentID, edge_length, mut_status = value.split(';')
        parentID = int(parentID)
        edge_length = float(edge_length)
        
        if mut_status == 'True':
            lambda_value = edge_length * mutation_constant
            mut_count = get_mutation_count(lambda_value)
            mut_IDs = ";".join(map(str, range(prev_mut + 1, prev_mut + mut_count + 1)))
            prev_mut += mut_count
            mut_dict[nodeID] = mut_IDs
        else:
            mut_dict[nodeID] = mut_dict[parentID]
    
    save_dict_to_file(mut_dict, out_file)
    return mut_dict, prev_mut + 1

def allocate_items(prob_dict, total_items):
  '''
  allocate exact total items based on a list of probability
  '''
  # Extract categories and their probabilities
  categories = list(prob_dict.keys())
  prob = list(prob_dict.values())
  # Calculate expected counts based on probabilities
  expected_counts = [p * total_items for p in prob]

  # Round the counts and calculate the initial sum
  rounded_counts = [round(count) for count in expected_counts]
  current_sum = sum(rounded_counts)
  #print("rounded counts", rounded_counts)
  # Adjust to ensure the sum of rounded counts is exactly 1000
  while current_sum != total_items:
    error = total_items - current_sum
    adjustment_indices = sorted(
      range(len(prob)), 
      key=lambda i: (expected_counts[i] - rounded_counts[i]), 
      reverse=(error > 0)
    )
    
    for i in adjustment_indices:
      if error > 0 and rounded_counts[i] < np.floor(expected_counts[i] + 0.5) + 1:
        rounded_counts[i] += 1
        current_sum += 1
      elif error < 0 and rounded_counts[i] > np.ceil(expected_counts[i] - 0.5) - 1:
        rounded_counts[i] -= 1
        current_sum -= 1
      if current_sum == total_items:
          break

  # Create a new dictionary to return the results
  allocated_items = {categories[i]: rounded_counts[i] for i in range(len(categories))}
  # Generate item numbers for each category
  item_numbers = {}
  start = 0
  for category, count in allocated_items.items():
    item_numbers[category] = np.array(list(range(start, start + count)))
    start += count
  return item_numbers

def distribute_snv_cells(cell_numbers, tree_file, out_file):
    tree_dict = OrderedDict()
    with open(tree_file, "r") as f:
        f.readline()  # Skip header
        for line in f:
            line = line.strip()
            if line:
                fields = line.split('\t')
                timepoint = float(fields[0])
                if timepoint == 1.0 or not timepoint.is_integer():
                    continue
                nodeID = int(fields[1])
                perc = float(fields[4])
                tree_dict.setdefault(timepoint, []).append(f"{nodeID};{perc}")
                
    snv_cell_dict = {}
    for i, (key, value) in enumerate(tree_dict.items()):
        prob_dict = {}
        total_perc = 0
        for node in value:
            nodeID, perc = node.split(';')
            prob_dict[str(int(key)) + "_" + nodeID] = float(perc)
            total_perc += float(perc)
        for node in prob_dict:
            prob_dict[node] = prob_dict[node]/total_perc
        cell_dict = allocate_items(prob_dict, cell_numbers[i])
        if i == 0:
            snv_cell_dict = cell_dict
        else:
            for node in cell_dict:
                cell_dict[node] += cell_numbers[i - 1]
        snv_cell_dict.update(cell_dict)
    for node in snv_cell_dict:
        temp = ";".join(str(x) for x in snv_cell_dict[node])
        snv_cell_dict[node] = temp
    save_dict_to_file(snv_cell_dict, out_file)
    return snv_cell_dict

def add_missing_data(matrix, missing_rate):
    n, m = len(matrix), len(matrix[0])
    missing_indices = random.sample(range(n * m), int(n * m * missing_rate))
    for idx in missing_indices:
        row, col = divmod(idx, m)
        matrix[row][col] = 3
    return matrix

def count_value(matrix, value):
    return sum(row.count(value) for row in matrix)

def add_false_pos_neg(matrix, alpha, beta):
    total_zeros = count_value(matrix, 0)
    total_ones = count_value(matrix, 1)
    
    fp_indices = random.sample(range(total_zeros), int(total_zeros * alpha))
    fn_indices = random.sample(range(total_ones), int(total_ones * beta))
    
    flat_matrix = [item for sublist in matrix for item in sublist]
    
    for idx in fp_indices:
        if flat_matrix[idx] == 0:
            flat_matrix[idx] = 1
    
    for idx in fn_indices:
        if flat_matrix[idx] == 1:
            flat_matrix[idx] = 0
    
    return [flat_matrix[i:i + len(matrix[0])] for i in range(0, len(flat_matrix), len(matrix[0]))]

def generate_mutation_matrix(mut_dict, snv_cell_dict, tree_file, num_cells, num_mutations, g_matrix_file):
    with open(tree_file, "r") as f:
        f.readline()  # Skip header
        edge_dict, node_dict = {}, {}
        for line in f:
            line = line.strip()
            if line:
                fields = line.split('\t')
                edge_dict[fields[1]] = Edge(fields[1], fields[2])
                node_dict[fields[1]] = Node(fields[1], fields[2], "NA")
    
    for parent, node in node_dict.items():
        if parent in edge_dict:
            node_dict[parent].c = edge_dict[parent].c
        else:
            node_dict[parent].c = "NA"
    
    snv_cell_mutations = {}
    for node, cells in snv_cell_dict.items():
        node_id = int(node.split('_')[1])
        parent_id = node_dict[str(node_id)].p
        mut_ids = mut_dict[node_id].split(";")
        parent_mut_ids = mut_dict[int(parent_id)].split(";")
        
        combined_mut_ids = set(mut_ids + parent_mut_ids)
        mut_dict[node_id] = ";".join(combined_mut_ids)
        
        for cell in cells.split(";"):
            snv_cell_mutations[cell] = list(combined_mut_ids)
    print("num_cells", num_cells, "num_mutations", num_mutations)
    g_matrix = init_matrix(num_cells, num_mutations)
    for cell_id in range(num_cells):
        if str(cell_id) in snv_cell_mutations:
            for mut_id in snv_cell_mutations[str(cell_id)]:
                g_matrix[cell_id][int(mut_id)] = 1
    save_matrix(g_matrix, g_matrix_file)
    return g_matrix

def vary_fp_fn_mr(g_matrix, snv_cell_dict, assigned_FP, assigned_FN, 
                  assigned_MR, d_matrix_file):
    timepoint_cells ={}
    for key, value in snv_cell_dict.items():
        if value == '':
            print("node with no cells")
            continue
        time = key.split("_")[0]
        cells = value.split(";")
        if time not in timepoint_cells:
            timepoint_cells[time] = cells
        else:
            timepoint_cells[time].extend(cells)
    final_d_matrix = []
    for index, (timepoint, cells) in enumerate(timepoint_cells.items()):
        g_timepoint = [g_matrix[int(cell)] for cell in cells]
        fp_rate, fn_rate, mr_rate = assigned_FP[index], assigned_FN[index], assigned_MR[index] 
        d_missing = add_missing_data(g_timepoint, mr_rate)
        d_matrix = add_false_pos_neg(d_missing, fp_rate, fn_rate)
        final_d_matrix.extend(d_matrix)
    save_matrix(final_d_matrix, d_matrix_file)

def sample_cells(num_timepoints):
    cells_options = [100, 300, 600, 1000]
    return [random.choice(cells_options) for _ in range(num_timepoints - 1)]

def sample_errors(mu, sd, times):
    # Define the lower bound (a) and upper bound (b) for truncation
    a, b = 0, np.inf
    # Calculate the parameters for the truncated normal distribution
    a, b = (a - mu) / sd, (b - mu) / sd
    # Generate x samples from the truncated normal distribution
    samples = truncnorm.rvs(a, b, loc=mu, scale=sd, size=times)
    return samples.tolist()

def sample_fp_fn_mr(num_timepoints, fp, fn, miss, fp_sd, fn_sd, miss_sd):
    fp_rates = sample_errors(fp, fp_sd, num_timepoints)
    fn_rates = sample_errors(fn, fn_sd, num_timepoints)
    missing_rates =  sample_errors(miss, miss_sd, num_timepoints)
    return fp_rates, fn_rates, missing_rates

def main():
    parser = argparse.ArgumentParser(description='This script generates mutation matrices with ground truth data.')
    parser.add_argument('-a', '--alpha', default=0.01, type=float, help='False positive rate.')
    parser.add_argument('-b', '--beta', default=0.2, type=float, help='False negative rate.')
    parser.add_argument('-m', '--missing_rate', default=0.2, type=float, help='Missing rate in G.')
    parser.add_argument('-t', '--timepoints', default=3, type=int, help='Number of timepoints.')
    parser.add_argument('-mc', '--mut_const', default=1.7, type=float, help='Mutation constant for Poisson distribution.')
    parser.add_argument('-f', '--tree_file', default="NA", help='The input tree structure file.')
    parser.add_argument('-P', '--prefix', default="NA", help='Prefix of output files.')
    parser.add_argument('-aSD', '--fp_sd', type = float, default=0.02, help='False positive rate variation')
    parser.add_argument('-bSD', '--fn_sd', type = float, default=0.1, help='False negative rate variation')
    parser.add_argument('-mSD', '--miss_sd',type = float, default=0.1, help='missing rate variation')


    args = parser.parse_args()
    
    if args.tree_file == "NA":
        print("""
        This generates the mutation matrix with the ground truth data. 
        Usage: python sim_par.py -a [alpha] -b [beta] -m [missing-rate] -t [num-timepoints] -mc [mut_const] -f [input-tree-file] -P [prefix-output-files]
            -a (--alpha)        False positive rate. [0.01]
            -b (--beta)         False negative rate. [0.2]
            -m (--missing_rate) Missing rate in G. [0.2]
            -t (--timepoints)   Number of timepoints. [3]
            -mc (--mut_const)   Mutation constant for Poisson distribution. [1.7]
            -f (--tree_file)    The input tree structure file. ["NA"]
            -P (--prefix)       Prefix of output files. 
        """)
        sys.exit(0)

    
    cell_numbers = sample_cells(args.timepoints)
    print(cell_numbers)
    mut_dict, mut_count = distribute_mutations(args.tree_file, args.mut_const, 
                                               args.prefix + ".mut.tsv")
    snv_cell_dict = distribute_snv_cells(cell_numbers, args.tree_file, 
                                                     args.prefix + ".SNVcell.tsv")
    g_matrix = generate_mutation_matrix(mut_dict, snv_cell_dict, args.tree_file, sum(cell_numbers), 
                             mut_count, args.prefix + ".G.tsv")
    assigned_FP, assigned_FN, assigned_MR = sample_fp_fn_mr(args.timepoints, args.alpha, args.beta,
                                                             args.missing_rate, args.fp_sd, args.fn_sd, args.miss_sd)
    vary_fp_fn_mr(g_matrix, snv_cell_dict, assigned_FP, assigned_FN, assigned_MR, args.prefix + ".D.tsv")

if __name__ == "__main__":
    main()