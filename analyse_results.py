import numpy as np
from dipy.io.pickles import load_pickle,save_pickle

labels = load_pickle('labels25000.pkl')
lists = load_pickle('lists25000.pkl')
C = load_pickle('C25000.pkl')
results = load_pickle('results25k.pkl')

metrics = np.zeros((len(results),7))

metric_names = results[(0,1)].keys()

metric_indices = [ i for (i,name) in enumerate(metric_names) if name != 'MatchedMapping' ]

for (i,r) in enumerate(results.keys()):
    for (m,k) in enumerate(metric_indices):
        #print i,r,m,k
        metrics[i,m] = results[r][metric_names[k]]
        
print metrics
