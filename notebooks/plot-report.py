import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from numpy.core.defchararray import mod
from pylab import style
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

def plot_length():
    max_lens = []
    recalls = []
    precisions = []
    f1s = []
    ovrecalls = []
    ivrecalls = []

    with open('fmm.log', 'r', encoding='utf8') as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            max_len = int(lines[idx].split(' = ')[-1])
            recall = float(lines[idx+1].split()[-1].strip())
            precision = float(lines[idx+2].split()[-1].strip())
            f1 = float(lines[idx+3].split()[-1].strip())
            ovrecall = float(lines[idx+6].split()[-1].strip())
            ivrecall = float(lines[idx+7].split()[-1].strip())
            max_lens.append(max_len)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            ovrecalls.append(ovrecall)
            ivrecalls.append(ivrecall)
            idx += 8
    plt.xlabel('maxlen')
    plt.ylabel('value')
    print(max_lens)
    plt.xticks(np.array(max_lens, dtype=np.int) - 1, max_lens)
    plt.plot(recalls, label='recall')
    plt.plot(precisions, label='precision')
    plt.plot(f1s, label='f1')
    plt.plot(ovrecalls, label='OOV Recall')
    plt.plot(ivrecalls, label='IV Recall')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('value-length.pdf')
    plt.show()

# plot_length()

def plot_report():
    max_lens = []
    recalls = []
    precisions = []
    f1s = []
    ovrecalls = []
    ivrecalls = []

    with open('../logs/maxent-report2.log', 'r', encoding='utf8') as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            max_len = int(lines[idx].split(' ')[-1])
            recall = float(lines[idx+1].split()[-1].strip())
            precision = float(lines[idx+2].split()[-1].strip())
            f1 = float(lines[idx+3].split()[-1].strip())
            ovrecall = float(lines[idx+6].split()[-1].strip())
            ivrecall = float(lines[idx+7].split()[-1].strip())
            max_lens.append(max_len)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            ovrecalls.append(ovrecall)
            ivrecalls.append(ivrecall)
            idx += 9


    plt.xlabel('iter')
    plt.ylabel('value')
    print(max_lens)
    plt.xticks(np.array(max_lens, dtype=np.int) - 1, max_lens)
    plt.plot(recalls, label='recall')
    plt.plot(precisions, label='precision')
    plt.plot(f1s, label='f1')
    plt.plot(ovrecalls, label='OOV Recall')
    plt.plot(ivrecalls, label='IV Recall')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('maxent-iter.pdf')
    plt.show()

# plot_report()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



def plot_best():
    plt.figure(figsize=(18, 10))
    # plt.style.use('seaborn-paper')
    plt.style.use('ggplot')
    models = ['MM', 'DAG', 'HMM', 'MaxEnt', 'CRF', 'pkuseg', 'thulac']
    f1 = [0.898, 0.902, 0.772, 0.836, 0.932, 0.880,  0.937]
    precision = [0.893, 0.881, 0.760, 0.841, 0.942, 0.867, 0.949]
    recall = [0.903, 0.924, 0.784, 0.832, 0.923, 0.894, 0.925]
    oovr = [0.0, 0.193, 0.436, 0.501, 0.593, 0.219, 0.791]
    ivr = [0.958, 0.968, 0.805, 0.852, 0.943, 0.935, 0.934]
    name_list = ['f1', 'precision', 'recall', 'IV Recall', 'OOV Recall']
    all_result = np.array([f1, precision, recall, ivr, oovr])
    mm = [0.898, 0.893, 0.903, 0.0, 0.958]
    dag = [0.902, 0.881, 0.924, 0.193, 0.968]
    hmm = [0.772, 0.76, 0.784, 0.436, 0.805]
    maxent = [0.836, 0.841, 0.832, 0.501, 0.852]
    crf = [0.932, 0.942, 0.923, 0.593, 0.943]
    pkuseg = [0.88, 0.867, 0.894, 0.219, 0.935]
    thulac = [0.937, 0.949, 0.925, 0.791, 0.934]
    cmap = get_cmap(len(models))
    colors = ['cyan', 'brown', 'b','orange', 'c','m', 'olive']
    x = list(range(len(mm)))
    total_width, n = 0.6, 6
    width = total_width / n
    for k in range(len(models)):
        if k == int(len(models) / 2):
            plt.bar(x, all_result[:, k], width=width, label=models[k], tick_label = name_list,ec='black',lw=.5)
        else:
            plt.bar(x, all_result[:, k], width=width, label=models[k], ec='black',lw=.5)
        for i in range(len(all_result[:, k])):
            plt.text(x[i], all_result[:, k][i]+0.01,'{:.1f}%'.format(all_result[:, k][i] * 100),ha='center',va='bottom',fontsize=10)
        for i in range(len(x)):
            x[i] += width

    plt.xlabel("测评指标",fontsize=12)
    plt.ylabel("分值",fontsize=12)
    plt.yticks(np.arange(0, 1.3, 0.1))
    plt.grid(b=True, which='major', axis='y', linestyle='--', linewidth=1)
    # plt.title("不同分词算法的对比",fontsize=15)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('best-plot.pdf')
    plt.show()


plot_best()