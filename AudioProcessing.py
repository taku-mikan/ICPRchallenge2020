# __feature__ : python2系の挙動をpython3系に変更するモジュール
# https://qiita.com/amedama/items/5e5a09b3a88dfd48198e
from __future__ import absolute_import, division, print_function

import os
import time

import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile
from tqdm import tqdm

"""
$ python preprocessing_T2.py --root D:\codes\dataset 
"""

class AudioProcessing():
    # mfccに関するわかりやすそうな記事
    # https://aidiary.hatenablog.com/entry/20120225/1330179868
    # https://qiita.com/tmtakashi_dist/items/eecb705ea48260db0b62
    def __init__(self,sample_rate,signal,frame_length_t=0.025,frame_stride_t=0.01,nfilt =64):
        
        self.sample_rate=sample_rate
        self.signal = signal
        self.frame_length_t=frame_length_t
        self.frame_stride_t=frame_stride_t
        self.signal_length_t=float(signal.shape[0]/sample_rate)
        self.frame_length=int(round(frame_length_t * sample_rate)) #number of samples
        self.frame_step=int(round(frame_stride_t * sample_rate))
        self.signal_length = signal.shape[0]
        self.nfilt=nfilt
        self.num_frames = int(np.ceil(float(np.abs(self.signal_length - self.frame_length)) / self.frame_step))
        self.pad_signal_length=self.num_frames * self.frame_step + self.frame_length
        self.NFFT=512 # fftのサンプル数(記事中のN)
        
    def calc_MFCC(self):
        # mfccへの変換手順
        # (1) 波形を適当な長さで分割し、窓関数をかけ、fftを行う
        pre_emphasis=0.97
        emphasized_signal=np.concatenate([self.signal[0,:].reshape([1,-1]),  self.signal[1:,:] - pre_emphasis * self.signal[:-1,:]], 0)
        z = np.zeros([self.pad_signal_length - self.signal_length,8])
        pad_signal = np.concatenate([emphasized_signal, z], 0)
        indices = np.tile(np.arange(0, self.frame_length), (self.num_frames, 1)) + np.tile(np.arange(0, self.num_frames * self.frame_step, self.frame_step), (self.frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        # (2) 窓処理を行なって、振幅スペクトルを求める
        # ハミング窓をかける
        frames=frames*np.hamming(self.frame_length).reshape(1,-1,1)
        frames=frames.transpose(0,2,1)
        mag_frames = np.absolute(np.fft.rfft(frames,self.NFFT)) # spec(記事中の)
        pow_frames = ((1.0 / self.NFFT) * ((mag_frames) ** 2)) # fscale(記事中の)

        # (3) メルフィルタバンクをかける
        filter_banks = np.dot(pow_frames, self.cal_fbank().T)
        # np.where : https://numpy.org/doc/stable/reference/generated/numpy.where.html
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)  # dB
        # np.transpose : https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
        # -->　配列の軸の順番を入れ替える
        filter_banks =filter_banks.transpose(0,2,1)

        # (3) 離散コサイン変換を行う --> やってない : 深層学習と合わないとかなんだとか
        
        return filter_banks
           
    def cal_fbank(self):
        # おそらくメルフィルタバンクなるものを作る関数
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.sample_rate / 2) / 700))  
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.nfilt + 2)  
        hz_points = (700 * (10**(mel_points / 2595) - 1)) 
        bin = np.floor((self.NFFT + 1) * hz_points / self.sample_rate)
        fbank = np.zeros((self.nfilt, int(np.floor(self.NFFT / 2 + 1))))
        for m in range(1, self.nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        return fbank


if __name__ == "__main__":

    import argparse
    """
    argparseの説明 : pythonでコマンド引数を取るときに用いる
    https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--ratio_step', type=float, default=0.25)
    parser.add_argument('--trimming', type=bool, default=False, choices=[True, False])
    parser.add_argument("--threshold", type=int, default=20)
    args = parser.parse_args()
    
    start = time.time()

    # data path -> root_pth
    root_pth = args.root
    
    # df <- annotations.csv
    df = pd.read_csv('annotations_sort.csv', header = 0)
    df_len=len(df) # データ数

    """
    フォルダの作成
    data
      |- audio 
           |- mfcc 
    """
    os.makedirs(os.path.join(root_pth, 'audio'), exist_ok=True)
    mfcc_path = (os.path.join(root_pth, 'audio', 'mfcc'))
    os.makedirs(mfcc_path,exist_ok=True)

    # 各種保存したい値のリストを作成
    count = 0
    pouring_or_shaking_list = []
    file_idx_list = []
    filling_type_list = []
    folder_count = [0]*9
    folder_count_detail = [[] for _ in range(9)]

    pbar = tqdm(total=df_len)
    save_size = 64
    threshold = args.threshold
    
    # indexで指定して1行ずつ処理する
    for fileidx in range(df_len):
        # pandas : df.iatの説明
        # https://note.nkmk.me/python-pandas-at-iat-loc-iloc/
        file_name = df.iat[fileidx, 2]
        folder_num = df.iat[fileidx, 0] # container_id
        start_time =  df.iat[fileidx, 9] # start
        end_time = df.iat[fileidx, 10] # end
        filling_type = df.iat[fileidx, 4] # filling_type:0~3(none pasta rice water)
        
        # python : rsplitの説明
        # https://note.nkmk.me/python-split-rsplit-splitlines-re/
        # s0_fi0_fu0_b0_l0_c2 -> s0_fi0_fu0_b0_l0_audio.wav
        audio_filename = file_name.rsplit("_", 1)[0] + '_audio.wav'

        audio_path = os.path.join(root_pth, str(folder_num), 'audio', audio_filename)
        # 377番目の音声データは飛ばす
        if audio_path == "./data/1/audio/s2_fi1_fu2_b1_l0_audio.wav" :
            continue

        # wavファイルの読み取り : scipy.io.wavfile ↓公式サイト
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
        # 返り値 : sample_rate==int signal==numpy array (N_samples, N_channels)
        sample_rate, signal = scipy.io.wavfile.read(audio_path)
        # sample_rate:44100, signal:(N, 8)(numpy.ndarray)
        
        # ここに他の前処理を加えればいいんじゃないかな
        # trimming
        # if args.trimming:
        #     if folder_num == 0:
        #         signal = signal.astype("float32")
        #     else :
        #         # numpyのcast 
        #         # https://note.nkmk.me/python-numpy-dtype-astype/
        #         signal = signal.astype("float32")
        #         # トリミング手法
        #         # https://librosa.org/doc/main/generated/librosa.effects.trim.html
        #         signal, _ = librosa.effects.trim(signal, top_db=threshold)

        signal = signal.astype("float32")
        if args.trimming:
            if filling_type in [1,2]:
                signal, _ = librosa.effects.trim(signal, top_db=threshold)
                # signal = signal.astype("int16")
        signal /= np.abs(signal).max() # 正規化
    
        ap = AudioProcessing(sample_rate,signal,nfilt=save_size)
        mfcc = ap.calc_MFCC()
        # mfcc : (N, 64, 8)(numpy.ndarray)
        mfcc_length=mfcc.shape[0] # N

        if mfcc_length < save_size:
            print("file {} is too short".format(fileidx))
        else:
            # このelse以下が何やってるのかよくわからん
            f_step=int(mfcc.shape[1]*args.ratio_step) # 64 * 0.25 = 16
            f_length=mfcc.shape[1] # 64

            # np.ceil : 小数点の切り上げ
            save_mfcc_num=int(np.ceil(float(np.abs(mfcc_length - save_size)) / f_step)) #000000.wavでは13
            folder_count_detail[folder_num-1].append(save_mfcc_num)

            for i in range(save_mfcc_num):
                count += 1
                tmp_mfcc = mfcc[i*f_step:save_size+i*f_step,: ,:] # (64, 64, 8)

                # start-endは後回し...
                # if start_time == -1:
                #     pouring_or_shaking_list.append(0)
                # elif start_time/ap.signal_length_t*mfcc_length<i*f_step+f_length*0.75 and end_time/ap.signal_length_t*mfcc_length>i*f_step+f_length*0.25:
                #     pouring_or_shaking_list.append(1) 
                # else:
                #     pouring_or_shaking_list.append(0)

                # container_id(folder_num)が1~6ならpouring, 7~9:shaking
                pouring = [1,2,3,4,5,6]
                shaking = [7,8,9]
                if folder_num in pouring :
                    pouring_or_shaking_list.append(1)
                elif folder_num in shaking:
                    pouring_or_shaking_list.append(0)
                else :
                    print("no container id")
                
                filling_type_list.append(filling_type)
                file_idx_list.append(fileidx)
                folder_count[folder_num-1] += 1
                
                np.save(os.path.join(mfcc_path, "{0:06d}".format(count)), tmp_mfcc)
                # 000001.npy 000002.npy ... 031796.npy
                
        pbar.update()

    np.save(os.path.join(root_pth, 'audio', 'pouring_or_shaking'), np.array(pouring_or_shaking_list) )
    np.save(os.path.join(root_pth, 'audio', 'filling_type'), np.array(filling_type_list))
    np.save(os.path.join(root_pth, 'audio', 'folder_count'), np.array(folder_count))
    np.save(os.path.join(root_pth, 'audio', 'folder_count_detail'), np.array(folder_count_detail))

    # pouring_or_shaking : 31796要素の0,1のnumpy配列
    # filling_type : 31796要素の0,1のnumpy配列
    # folder_count : [6134, 4534, 4386, 3994, 4342, 5382, 1100, 1001, 923]
    # folder_count_detail : (9, 84)次元のnumpy配列

    elapsed_time = time.time() - start
    print("elapsed_time:{}".format(elapsed_time) + "sec")
