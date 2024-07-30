import argparse
import pandas as pd
import numpy as np

def getInput(Dfile, cellsFile, outF):
  # get BnpC_ext input
  D = np.genfromtxt(Dfile, delimiter='\t')
  cell_time = [-1] * D.shape[0] # time for each cell, starting from 0
  f = open(cellsFile, "r")
  time2id = {}
  id = 0
  for line in f.readlines():
    temp = line.rstrip().split()
    time = temp[0].split("_")[0]
    if time not in time2id:
      time2id[time] = id
      id += 1
    cells = temp[1].split(";")
    for c in cells:
      cell_time[int(c)] = time2id[time]
  f.close()
  time_col = np.array(cell_time)
  D_time = np.column_stack((D, time_col))
  np.savetxt(outF, D_time, delimiter='\t', fmt='%d')

def getClusters(mutFile, cellsFile, outF):
  mut2node= {} #{mutation: [node1, node2 ...]}
  f = open(mutFile, "r")
  for line in f.readlines():
    temp = line.rstrip().split()
    mut = temp[1]
    if mut not in mut2node:
      mut2node[mut] = [temp[0]]
    else:
      mut2node[mut].append(temp[0])
  f.close()
  cluster2cell = {}
  node2cell = {}
  f = open(cellsFile, "r")
  for line in f.readlines():
    temp = line.rstrip().split()
    node = temp[0].split("_")[1]
    node2cell[node] = []
    cells = temp[1].split(";")
    for c in cells:
      node2cell[node].append(c)
  f.close()
  counter = 0
  print(mut2node)
  for key, nodes in mut2node.items():
    cells = []
    for node in nodes:
      if node not in node2cell:
        continue
      cells += (node2cell[node])
    if cells:
      cluster2cell[counter] = cells
      counter += 1
  f = open(outF, "w")
  for cluster, cells in cluster2cell.items():
    f.write(str(cluster) + "\t" + ";".join(cells) + "\n")
  f.close()

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--wd', help = "working directory", default="./")
args = parser.parse_args()
getInput(args.wd + "/test.D.csv", args.wd + "/test.SNVcell.csv", args.wd + "/BnpC_ext_input.tsv")
getClusters(args.wd + "/test.mut.csv", args.wd + "/test.SNVcell.csv", args.wd + "/true_cluster.tsv")
