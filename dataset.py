import os

import torch
import torchaudio
import torchvision
import numpy as np
from num2words import num2words
import re
import editdistance


class MyDataset:
    letters = [' ', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', \
               'у', 'ф', 'ц', 'х', 'ш', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']

    def __init__(self, root_dir,label_path,subset,modality,audio_transform=None,
                 video_transform=None,rate_ratio=640,):
        self.root = root_dir
        self.label_path= label_path
        self.subset = subset
        self.modality = modality
        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.rate_ratio = rate_ratio

        # self.vid_pad = 
        # self.txt_pad = 200


        self.list_files = self.load_list(label_path)
        # self.load_list(label_path)
        # print(self.list_files)

    def load_list(self, label_path):
        print('here')
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            # print(path_count_label)
            dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            txt_path = os.path.join(self.root, dataset_name,rel_path.replace('video', 'text').replace('mp4', 'txt'))
            content = self.load_anno(txt_path)
            # print(MyDataset.arr2txt(content, 1))
            paths_counts_labels.append(
                (
                    dataset_name,
                    rel_path,
                    int(input_length),
                    torch.tensor([int(_) for _ in content]),
                )
            )
        return paths_counts_labels
    

    def load_anno(self, name):
        with open(name, 'r') as f:
            line = f.readlines()[0].replace('-', ' ')
            line = re.sub(r'[^0-9а-яА-Я ]', '', line)
            line = line.strip().split(' ')
            for i, char in enumerate(line):
                if char.isdigit():
                    line[i] = num2words(char, lang='ru')
            # print(line)
        return MyDataset.txt2arr(' '.join(line), 1)
    

    def load_video(self,path):
        vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
        print('VIDEO SHAPE: ', vid.shape)
        print('NEEDED SHAPE: ', vid.permute(3, 0, 1, 2).shape)
        vid = vid.permute((0, 3, 1, 2))
        # vid = vid.permute(3, 0, 1, 2)
        print('PERMUTED SHAPE: ', vid.shape)
        

        return vid
    
    def load_audio(path):
        waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
        return waveform.transpose(1, 0)


    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id = self.list_files[idx]
        path = os.path.join(self.root, dataset_name, rel_path)
        if self.modality == "video":
            video = self.load_video(path)
            video = self.video_transform(video)
            # video = video.permute(3, 0, 1, 2)

            print('SHAPE AFTER TRANSFORM: ', video.shape)
            # video = video.permute(1,0,2,3)
            # print('CHECK SHAPE: ', video.shape)



            # video = self._padding(video, self.vid_pad)
            # token_id = self._padding(token_id, self.txt_pad)
            return {"vid": video, "txt": token_id}
            # return {'vid': torch.FloatTensor(video), 
            # 'txt': torch.LongTensor(token_id),
            # 'txt_len': len(token_id),
            # 'vid_len': input_length}


        # elif self.modality == "audiovisual":
        #     video = self.load_video(path)
        #     audio = self.load_audio(path)
        #     audio = self.cut_or_pad(audio, len(video) * self.rate_ratio)
        #     video = self.video_transform(video)
        #     audio = self.audio_transform(audio)
        #     return {"video": video, "audio": audio, "target": token_id}


    def __len__(self):
        return len(self.list_files)
    

    # def cut_or_pad(self, data, size, dim=0):
    #     """
    #     Pads or trims the data along a dimension.
    #     """
    #     if data.size(dim) < size:
    #         padding = size - data.size(dim)
    #         data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
    #         size = data.size(dim)
    #     elif data.size(dim) > size:
    #         data = data[:size]
    #     assert data.size(dim) == size
    #     return data


    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)


    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)
        
    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(MyDataset.letters[n - start])     
        return ''.join(txt).strip()
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):                
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])                
            pre = n
        return ''.join(txt).strip()
    
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer
    


if __name__ == "__main__":
    dataset = MyDataset(
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_",
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_train_transcript_lengths_seg24s_0to100_5000units.csv",
        "train",
        "video"
    )
    # dataset.load_anno()
    # dataset
