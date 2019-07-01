import sys
import numpy as np

fn = sys.argv[2]

thresholds = []
with open(fn) as fp:
    fp.readline()
    for e, line in enumerate(fp):
        line = line.strip().split('\t')
        assert(int(line[0])==e)
        thresholds.append(float(line[1]))

fp = open(sys.argv[1])
fp.readline()
ACC = []
AUC = []
Ps  = []

rACCs = []
# only look at up to ACC@10
for i in range(min(13,len(thresholds))):
    rACCs.append([])

for line in fp:
    line = line.strip().split('\t')
    prob = float(line[3])
    rad = 0
    acc = int(line[5])
    if acc == 1:
        ACC.append(1.)
        Ps.append(prob)
        
        for idx in range(len(rACCs)):
            tau = thresholds[idx]
            if prob > tau:
                rACCs[idx].append(1.)
                rad = idx
            else:
                rACCs[idx].append(0.)
    else:
        ACC.append(0.)
        for idx in range(len(rACCs)):
            rACCs[idx].append(0.)
    AUC.append(rad)

print('mean/median = {:.3f}/{:.3f}'.format(np.mean(AUC), np.median(AUC)))
print('ACC = {:.3f}, # = {}'.format(np.mean(ACC), len(ACC)))
print('Prob: 25, 50, 75, 100 =', np.percentile(Ps, 25), np.percentile(Ps, 50), np.percentile(Ps, 75), np.max(Ps))
for idx in range(len(rACCs)):
    print('r = {:3d}, ACC = {:.3f}, threshold = {}'.format(idx, np.mean(rACCs[idx]), thresholds[idx]))


