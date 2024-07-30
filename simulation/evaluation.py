from sklearn.metrics.cluster import v_measure_score
import sys 
import numpy as np

trueF = sys.argv[1]
predF = sys.argv[2]

f = open(predF, "r")
line = f.readlines()[1].rstrip().split()
pred = [int(x) for x in line[2:]]
f.close()

true = [-1] * len(pred)
true = np.array(true)
f = open(trueF, "r")
for line in f.readlines():
  label = int(line.rstrip().split()[0])
  cells = [int(x) for x in line.rstrip().split()[1].split(";")]
  true[cells] = label

v_measure = v_measure_score(true, pred)
print("v measure", v_measure)
