import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

"""setup"""
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
# --> pytorch : CNNのlossが毎回変わることを防ぐ == 決定論的な振る舞いを指定
# https://qiita.com/chat-flip/items/c2e983b7f30ef10b91f6

torch.backends.cudnn.benchmark = False
# --> これは何(公式サイトは以下)
# https://pytorch.org/docs/stable/backends.html

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,root_pth,test=False,transform = None):
        # クラス数
        class_num=4
        # 各データへのpathの設定
        self.audio_pth = os.path.join(root_pth, 'audio', 'mfcc')
        filling_type = np.load(os.path.join(root_pth, 'audio', 'filling_type.npy'))
        pouring_or_shaking = np.load(os.path.join(root_pth,  'audio', 'pouring_or_shaking.npy'))

        # self.label : 31796要素分の(0,1,2,3)のnumpy配列 --> クラスかな？？
        self.label = filling_type * pouring_or_shaking

        self.is_test = test # testセットかどうかの保存変数
        self.each_class_size = [] # それぞれのクラスに属するデータの数

        # 各クラスに属するデータ数を保存
        for i in range(class_num):
            self.each_class_size.append(np.count_nonzero(self.label==i))
 
        # mfccフォルダのデータを一つずつ確認しmax. minを更新する(min maxが何を表すかは謎)
        mx = 0
        mn = 1000
        for idx in range(self.label.shape[0]):
            data=np.load(os.path.join(self.audio_pth, "{0:06d}".format(idx+1) + '.npy'), allow_pickle=True)
            # data === 000001.npy 0000002.npy ... 
            tmp_max = np.max(data)
            tmp_min = np.min(data)
            if mx < tmp_max:
                mx = tmp_max
            if mn > tmp_min:
                mn = tmp_min
        self.mn = mn
        self.mx = mx
            
    def __len__(self):
        # self.label.shape = (31769, )
        return self.label.shape[0]
    
    def __getitem__(self, idx):
        """
        index(idx)を受け取り、学習(self.is_test=False)ならlabelと一緒に,
        testセットならdataだけ(labelは-1)を返す
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lbl = -1

        if self.is_test is False:
            lbl = self.label[idx]
        data=np.load(os.path.join(self.audio_pth, "{0:06d}".format(idx+1) + '.npy'), allow_pickle=True)
        # 正規化
        data = (data-self.mn)/(self.mx-self.mn)
        data = data.transpose(2,0,1) # pytorchの入力に合わせているのかと
        data = torch.from_numpy(data.astype(np.float32))
        return data , lbl
            
    def get_each_class_size(self):
        return np.array(self.each_class_size)



class Padding(object):
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, sample, pred):
        #np.clip(pred, 0,1,out=pred)
        sample_len, input_dim = sample.shape
        #for i in range(sample_len):
        #    sample[i, :] *= pred[i]

        if sample_len >= self.seq_len:
            features = sample[:self.seq_len, :]
            return features
        else:
            start_seq = np.random.randint(0, self.seq_len - sample_len+1)
            #ini=[1]+[0]*(input_dim-1)
            ini=[0]*(input_dim)
            features = np.full((self.seq_len, input_dim),ini, dtype = float)
            features[start_seq:start_seq+sample_len, :] = sample
            return features

    


class MyLSTMDataset(torch.utils.data.Dataset):

    def __init__(self,root_pth,test=False, transform=None, padding_size=100):
        class_num=3 # クラス数 : 0,50,90%
        self.mid_pth = os.path.join(root_pth, 'T2_mid')
        self.pred_pth = os.path.join(root_pth, 'T2_pred')

        df = pd.read_csv(os.path.join('.', 'annotations_sort.csv'), header=0, usecols=[5])
        self.label = df['filling_level'].values # Task1におけるlabel
        self.is_test=test # testデータか否か
        self.each_class_size = [] # それぞれのラベルの数を保存 : sumとの違いはなんや
        self.each_class_sum = [0]*class_num # それぞれのファイルの各ラベル

        for i in range(class_num):
            self.each_class_size.append(np.count_nonzero(self.label==i))

        mx=0
        mn=1000
        len_mx = 0
        
        # index指定して1ファイルずつ処理する
        for idx in range(self.label.shape[0]):
            # 音声ファイルの読み込み
            data = np.load(os.path.join(self.mid_pth, "{0:06d}".format(idx+1) + '.npy'), allow_pickle=True)
            self.each_class_sum[self.label[idx]] += data.shape[0]
            
            if data.shape[0] > len_mx:
                len_mx = data.shape[0]

            tmp_max = np.max(data)
            tmp_min = np.min(data)

            if mx < tmp_max:
                mx = tmp_max
            if mn > tmp_min:
                mn = tmp_min

        self.mn = mn # 全体の中のmin
        self.mx = mx # 全体の中のmax
        self.pad = Padding(padding_size)
        print("max_langth : ", len_mx)
            
    def __len__(self):
        """
        データ数を返す関数
        self.label = 全データのlabel
        """
        return self.label.shape[0]
    
    def __getitem__(self, idx):
        """
        trainデータなら、dataとlabelをセットで、
        testデータなら、dataのみを(label=-1)返す関数
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lbl = -1

        if self.is_test is False:
            lbl = self.label[idx]

        data = np.load(os.path.join(self.mid_pth, "{0:06d}".format(idx+1) + '.npy'), allow_pickle=True)
        pred = np.load(os.path.join(self.pred_pth, "{0:06d}".format(idx+1) + '.npy'), allow_pickle=True)
        data = (data-self.mn)/(self.mx-self.mn) # 正規化
        data = self.pad(data, pred)

        # np.clip(data, 0,1,out=data)
        data=torch.from_numpy(data.astype(np.float32))
        return data , lbl
            
    def get_each_class_size(self):
        """
        それぞれのクラスに属するデータの数を返す関数(numpy配列で)
        """
        return np.array(self.each_class_size)

    def get_each_class_avg_len(self):
        """
        each~~~size,numの違いがわからん!!
        """
        each_class_avg_len =  np.array(self.each_class_sum)/np.array(self.each_class_size)
        all_class_avg_len = np.sum(np.array(self.each_class_sum))/np.sum(np.array(self.each_class_size))
        return each_class_avg_len, all_class_avg_len

    
if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, default='./data')
    args = parser.parse_args()
    root_pth = args.root

    mydataset = MyDataset(root_pth)
    
    data, lbl = mydataset[150]
    print(mydataset.mn, ' ', mydataset.mx)
    
    # mydataset=MyDataset(root_pth)
    mylstmdataset = MyLSTMDataset(root_pth)
    data, lbl = mylstmdataset[150]
    print(data, ' ', lbl)
    print(torch.max(data))
    print(torch.min(data))
    print(mylstmdataset.mn, ' ', mylstmdataset.mx)
    print(mylstmdataset.get_each_class_avg_len())

    """
    -313.07119549054045   194.19187653405487
953
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])   2
tensor(0.3164)
tensor(0.)
-1.1948369   57.464638
(array([13.05555556, 39.70833333, 65.85416667]), 46.50877192982456)
    """
