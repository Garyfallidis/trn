import numpy as np

from dipy.io.pickles import load_pickle,save_pickle

from scipy.stats import describe

import csv

C_sizes = ['10000','25000','50000','100000']

for i,s in enumerate(C_sizes):
    C=load_pickle('C'+s+'.pkl')
    stats=[(len(C[k]),sum([C[k][j]['N'] for j in C[k].keys()])) for k in C.keys()]
    print np.mean(np.array(stats),)
    
stop

sizes = ['10k','25k','50k','100k']

metrics = ['Purity', 'RandomAccuracy', 'PairsConcordancy', 'Completeness', 'Correctness', 'MatchedAgreement', 'MatchedKappa']

with open('tmp.csv', 'wb') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)

    #print f, '"Purity",,"Purity","Random Accuracy","Pairs Concordancy","Completeness","Correctness","Matched Agreement","Matched Kappa"\n'
    writer.writerow([None,None]+metrics)

    alldata = []

    for size in sizes:
    
        results=load_pickle('results'+size+'.pkl')
        
        keys = results.keys()
        
        table = np.zeros((len(results), len(metrics)))
        
        for i,k in enumerate(keys):
            r = results[k]
            for j,m in enumerate(metrics):
                table[i,j] = r[m]

        alldata = alldata + [table]
                
        d = describe(table,axis=0)
        
        #size = d[0]
        type = ['Min','Max','Mean','s.d.']
        #print >> f, size,'", Min,"'
        mins = (None,type[0])+tuple(d[1][0])
        maxs = (None,type[1])+tuple(d[1][0])
        means = (size,type[2])+tuple(d[2])
        sds = (None,type[3])+tuple(np.sqrt(d[3]))

        tab = np.vstack((means,sds))
        
        writer.writerows(tab)
        writer.writerow((None,))

f.close()

import matplotlib.pyplot as plt

fig = plt.figure()

for j, met in enumerate(metrics):

    ax = [None,None,None,None]
    '''
    for i,data in enumerate(alldata):
        if i == 0:
            ax[i] = fig.add_subplot(7,4,4*j+i+1)
        else:
            ax[i] = fig.add_subplot(7,4,4*j+i+1,sharex=ax[0],sharey=ax[0])
            
        n, bins, patches = ax[i].hist(data[:,j], normed = True, facecolor='green', alpha=0.75)
    '''
    for i,data in enumerate(alldata):

        if i == 0:
            ax[i] = fig.add_subplot(4,7,j+7*i+1)
        else:
            ax[i] = fig.add_subplot(4,7,j+7*i+1,sharex=ax[0],sharey=ax[0])
            
        n, bins, patches = ax[i].hist(data[:,j], normed = True, facecolor='green', alpha=0.75)
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])

plt.show()