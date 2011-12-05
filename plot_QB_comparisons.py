
import numpy as np
import matplotlib.pyplot as plt
    
N = 2
Means = (0.4690955208544213,0.52516460626048289)
Stds =   (0.02635549381315298,0.049120754359623708)

ind = np.array([0.5,1.5])#np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, Means, width, color='r', yerr=Stds, capsize=10,linewidth=None)
    
# add some
ax.set_ylabel('mean')
ax.set_title('TC comparisons between real subjects')
ax.set_xticks(ind+width/2)
ax.set_xticklabels(('TC10', 'TC100'))

#ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )    
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)
plt.savefig('/home/eg309/Documents/didaktoriko/last_figures/TC_comparisons_diff_subjects.png')
#plt.show()
