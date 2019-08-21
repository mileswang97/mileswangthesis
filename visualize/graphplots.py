#library for standardizing pyplot data visualizations on distributions

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from collections import defaultdict


# should be placed higher, called in graph plots to save numpy array information
def createExcelSheets():
    pass

def createHistogram():
    pass

def createBarChart(data1, labels1, labels2, title, study_count, path):
    fix, ax = plt.subplots()
    ind = np.arange(data1.shape[0])
    width = 0.15

    data1 = data1.T 
    # for per study purposes
    data1 = data1/study_count

    bar_dict = defaultdict()
    for index in range(data1.shape[0]):
        try:
            # need to change color variable later
            bar_dict[index] = ax.bar(ind + width*index, tuple(data1[:][index]), width, color='red')
        except:
            print('noooo')

    ax.set_xticks(ind + width/2)
    ax.set_xtickslabels(labels1, fontsize=8)
    ax.legend(tuple([i[0] for i in bar_dict.values()]), labels2)
    ax.autoscale_view()
    plt.title(str(title), y=1.05)
    print('saving bar chart ...')

    try:
        plt.savefig(path, dpi=300, format='jpg', bbox_inches='tight')
    except:
        os.makedirs(path, 0o777)
        plt.savefig(path, dpi=300, format='jpg', bbox_inches='tight')

    plt.close(fig='all')
    return

def createPieChart():
    pass

# i suspect this will be used for graphing progress in measuring accuracy, auc etc.
def createLineChart():
    pass

