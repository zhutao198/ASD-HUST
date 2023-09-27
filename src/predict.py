import os
import numpy as np
import torch
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn import metrics

from model import ECGNet
# from model import ECGNet_12 as ECGNet
import scipy.io as sio

cur_path = os.path.split(os.path.realpath(__file__))[0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dict = {0: '正常', 1: '房缺'}
model_path="k10"
class SASDModel:
    def __init__(self):
        self.device = torch.device("cuda:0")
        model_list = [i for i in os.listdir('../model/{}'.format(model_path)) if 'bestf1' in i]
        self.load_mlist = [self.load_model('../model/{}/{}'.format(model_path,i)) for i in model_list]

    def load_model(self, model_name):
        m = ECGNet(num_classes=2).to(self.device)
        checkpoint = torch.load(model_name, map_location={'cuda:2': 'cuda:0', 'cuda:1': 'cuda:0', 'cuda:3': 'cuda:0',
                                                          'cuda:4': 'cuda:0','cuda:5': 'cuda:0','cuda:6': 'cuda:0',
                                                          'cuda:7': 'cuda:0','cuda:8': 'cuda:0'})
        m.load_state_dict(checkpoint['model'])
        m.eval()
        return m

    def predict(self, data):
        data = data.reshape((1, 8, 5000))
        data = torch.from_numpy(data).float().to(self.device)
        pavg = sum([i(data).cpu().detach().numpy() for i in self.load_mlist]) / 10
        # pavg = self.load_mlist[0](data).cpu().detach().numpy()
        cls = np.array(softmax(pavg, axis=1))
        # cls = np.argmax(pavg)
        return cls

def get_data(sasd_data_path='/home/workspace/ml_project/ASD-HUST/data/sasd',
             normal_data_path='/home/workspace/ml_project/ASD-HUST/data/normal'):
    def _get_data(data_path):
        _data = []
        f_list = [i for i in os.listdir(data_path) if i.endswith('.mat')]
        for f in tqdm.tqdm(f_list):
            f_path = os.path.join(data_path, f)

            try:
                fdata = sio.loadmat(f_path)
                data = fdata['ecg']
                if data.shape[1] != 5000:
                    print(f'{f} length is {data.shape[0]}.')
                    _data.append(data[:, -5000:])
                else:
                    _data.append(data)
            except:
                print(f'{f} read error.')
                continue
        _data = np.array(_data)
        return _data
    data1=_get_data(sasd_data_path)
    label1 = np.array(1*np.ones((data1.shape[0])))
    data2=_get_data(normal_data_path)
    label2 = np.array(0*np.ones((data2.shape[0])))
    _data=np.concatenate([data1,data2],axis=0)
    _label=np.concatenate([label1,label2],axis=0)
    return _data,_label.astype(np.int16)


def draw_matrix(cf_matrix, save_path, labels):
    def _heatmap(matrix, x, y, title, fmt):
        ax = sns.heatmap(matrix,
                         xticklabels=labels,
                         yticklabels=labels,
                         annot=True,
                         fmt=fmt,
                         ax=axes[x][y],
                         cmap='Blues')
        ax.title.set_text(title)
        ax.set_xlabel("预测")
        ax.set_ylabel("标签")
        return ax

    sns.set_theme(font='SimHei', font_scale=1.5, color_codes=True)
    figure, axes = plt.subplots(2, 3, figsize=(16 * 1.75, 16))
    _heatmap(cf_matrix, 0, 0, '混淆矩阵', 'g')
    _heatmap(cf_matrix / np.sum(cf_matrix), 1, 0, '混淆矩阵(%)', '.1%')
    # 精准矩阵，列和为1
    sum_pred = np.expand_dims(np.sum(cf_matrix, axis=0), axis=0)
    precision_matrix = cf_matrix / sum_pred
    _heatmap(precision_matrix, 0, 1, '精确度', '.1%')
    # 召回矩阵，行和为1
    sum_true = np.expand_dims(np.sum(cf_matrix, axis=1), axis=1)
    recall_matrix = cf_matrix / sum_true
    _heatmap(recall_matrix, 1, 1, '召回率', '.1%')
    # F1矩阵
    a = 2 * precision_matrix * recall_matrix
    b = precision_matrix + recall_matrix
    f1_matrix = np.divide(a, b, out=np.zeros_like(a), where=(b != 0))
    _heatmap(f1_matrix, 0, 2, 'F1值', '.1%')
    # 绘制5张图
    plt.autoscale(enable=False)
    plt.savefig(save_path)
    plt.close()


def get_metrics(matrix):
    cls = matrix.shape[0]
    tp = np.array([matrix[i, i] for i in range(cls)])
    fp = np.sum(matrix, axis=0) - tp
    fn = np.sum(matrix, axis=1) - tp
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    ans = np.array([tp, fp, fn, pre, rec, f1], dtype=np.float16)
    return ans.T

def draw_roc(probs, labels, save_path,nnum):
    """
    :param probs: 概率数组，n条数据的c类概率，shape=(n,c)
    :param labels: 标签数组或列表，n条数据的类别，shape=(n,)
    :return: 无返回值，但会生成各个类别的roc曲线及此类的auc值
    """
    for cls in range(max(labels) + 1):
        scores = probs[:, cls]
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=cls)
        pos = np.argmax(tpr ** 2 + (1 - fpr) ** 2)
        auc = metrics.auc(fpr, tpr)
        s = 'Best thresholds ({:.2f})\nTPR = {:.2f}\nFPR = {:.2f}'.format(thresholds[pos],tpr[pos],fpr[pos])
        plt.scatter(fpr[pos], tpr[pos], s=15, color='k',label=s,zorder=2)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc,zorder=1)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',zorder=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve of class {}'.format(cls))
        plt.legend(loc="lower right")
        plt.savefig('{}/roc_of_{}_{}.png'.format(save_path,nnum, cls))
        plt.close()

def get_matrix(p, t):
    num_classes = 2
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(p)):
        matrix[t[i], p[i]] += 1
    return matrix


def get_seed_acc(y_probs,y_list):
    matrix = get_matrix(np.argmax(y_probs,axis=1),y_list)
    metrics = get_metrics(matrix)
    num=y_list.shape[0]
    save_path = '/home/workspace/ml_project/ASD-HUST/out/matrix_{}.png'.format(model_path)
    draw_matrix(matrix, save_path,['正常', '房缺'])
    draw_roc(y_probs,y_list,'/home/workspace/ml_project/ASD-HUST/out',num)
    print(metrics[:, 3:])
# Xt,Yt=get_data()
if __name__ == '__main__':
    m = SASDModel()
    print('All models have been loaded.')
    Xt,Yt=get_data()
    # Xt = Xt.transpose(0, 2, 1)
    Pt=np.zeros((Yt.size,2))
    for i,data in enumerate(Xt):
        pre = m.predict(data)
        Pt[i]=pre
        print(f"{pre}/{Yt[i]}")
    get_seed_acc(Pt, Yt)
    with open('/home/workspace/ml_project/ASD-HUST/out/y_label.csv','w') as fd:
        for _y in Yt:
            fd.write(f'{_y}\n')
    with open('/home/workspace/ml_project/ASD-HUST/out/y_score.csv','w') as fd:
        for _p in Pt:
            fd.write(f'{_p[0]},{_p[1]}\n')
    print("ok")
