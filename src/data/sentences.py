import os

import torch
# import torchaudio
import torchvision
import numpy as np
from num2words import num2words
import re
import editdistance
import matplotlib.pyplot as plt
import cv2


from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.transforms import StatefulRandomHorizontalFlip
import random

class MyDataset:
    # characters = [' ', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', \
    #            'у', 'ф', 'ц', 'х', 'ш', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
    # characters = [char for char in ' абвгдежзийклмнопрстуфхцчшщъыьэюя']
    characters = [char for char in ' абвгдежзийклмнопрстуфхцчшщъыьэюя']
    def __init__(self, root_dir,label_path,subset,modality,audio_transform=None,
                 video_transform=None,rate_ratio=640,):
        self.root = root_dir
        self.label_path= label_path
        self.subset = subset
        self.modality = modality
        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.rate_ratio = rate_ratio

        self.vid_pad = 400
        self.txt_pad = 400


        self.list_files = self.load_list(label_path)[:100]
        # self.list_files = self.load_list(label_path)
        # self.list_files = self.list_files[:len(self.list_files)//3]


        # self.load_list(label_path)
        # ##print(self.list_files)

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            txt_path = os.path.join(self.root, dataset_name,rel_path.replace('video', 'text').replace('mp4', 'txt'))
            content = token_id
            paths_counts_labels.append(
                (
                    dataset_name,
                    rel_path,
                    int(input_length),
                    content,
                )
            )
        return paths_counts_labels
    

    def load_anno(self, name):
        with open(name, 'r') as f:
            line = f.readlines()[0].replace('-', ' ')
            # print(line)
            line = re.sub(r'[^0-9а-яА-Я ]', '', line)
            line = line.strip().split(' ')
            for i, char in enumerate(line):
                if char.isdigit():
                    line[i] = num2words(char, lang='ru')
        return MyDataset.txt2arr(' '.join(line), 1)

    

    def load_video(self,path):
        vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
        vid = vid.permute((0, 3, 1, 2))
        return vid
    

    def __getitem__(self, idx):
        # print("LIST FILES: ", self.list_files[idx])
        dataset_name, rel_path, input_length, _ = self.list_files[idx] # for vmeste

        path = os.path.join(self.root, dataset_name, rel_path)
        if self.modality == "video":
            video = self.load_video(path) # for vmeste
            video = self.video_transform(video)
            # video = (video - video.mean()) / video.std()
            length_video = video.shape[0]
            # print("PATH TXT: ", path.replace("video", "text").replace("mp4", "txt"))
            labels = self.load_anno(path.replace("video", "text").replace("mp4", "txt"))
            # print("LABELS: ", labels)
            return video, torch.tensor(labels, dtype=torch.int), length_video, len(labels)
            # return {"video": video, "labels": labels, "length": length}S


    def __len__(self):
        return len(self.list_files)


    def _padding(self, array, length, pad_val):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)


    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.characters.index(c) + start)
        return np.array(arr)
        

    @staticmethod
    def arr2txt(arr, start):
        txt = []
        # ##print(arr.detach().cpu().numpy(), type(arr.detach().cpu().numpy()))
        # arr = [t[t != -1] for t in arr]
        arr = arr[arr != -1]
        # ##print(arr)
        for n in arr:
            if(int(n) >= start):
                txt.append(MyDataset.characters[int(n) - start])     
        return ''.join(txt).strip()
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        # pre = -1
        pre = 0
        txt = []
        # print("ARRAY: ",arr)
        for n in arr:
            if(pre != n and n >= start):                
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.characters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.characters[n - start])                
            pre = n
        
        return ''.join(txt).strip()
    
    # @staticmethod
    # def wer(predict, truth):        
    #     word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
    #     # print("WORD PAIRS: ",word_pairs)
    #     wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
    #     return np.array(wer)
        
    # @staticmethod
    # def cer(predict, truth):        
    #     cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
    #     return np.array(cer)

    @staticmethod
    def wer(predict, truth): 
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [editdistance.eval(p[0], p[1]) / len(p[1]) for p in word_pairs]
        return np.array(wer)

        
    @staticmethod
    def cer(predict, truth):        
        cer = [editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return np.array(cer)
    

def build_word_list(directory, num_words, seed):
    random.seed(seed)
    words = os.listdir(directory)
    words.sort()
    random.shuffle(words)
    words = words[:num_words]
    return words


class MyLRWDataset(Dataset):
    characters = [char for char in ' abcdefghijklmnopqrstuvwxyz']

    def __init__(self, path, num_words=500, in_channels=1, mode="train", augmentations=False, estimate_pose=False, seed=42, query=None):
        self.seed = seed
        self.num_words = num_words
        self.in_channels = in_channels
        self.query = query
        self.augmentation = augmentations if mode == 'train' else False
        self.poses = None
        # if estimate_pose == False:
        #     self.poses = self.head_poses(mode, query)
        self.video_paths, self.files, self.labels, self.words = self.build_file_list(path, mode)
        self.estimate_pose = estimate_pose

    # def head_poses(self, mode, query):
    #     poses = {}
    #     yaw_file = open(f"data/preprocess/lrw/{mode}.txt", "r")
    #     content = yaw_file.read()
    #     for line in content.splitlines():
    #         file, yaw = line.split(",")
    #         yaw = float(yaw)
    #         if query == None or (query[0] <= yaw and query[1] > yaw):
    #             poses[file] = yaw
    #     return poses

    def build_file_list(self, directory, mode):
        words = build_word_list(directory, self.num_words, seed=self.seed)
        # print(words)
        paths = []
        file_list = []
        labels = []
        for i, word in enumerate(words):
            dirpath = directory + "/{}/{}".format(word, mode)
            files = os.listdir(dirpath)
            for file in files:
                if file.endswith("mp4"):
                    if self.poses != None and file not in self.poses:
                        continue
                    path = dirpath + "/{}".format(file)
                    file_list.append(file)
                    paths.append(path)
                    labels.append(i)

        return paths, file_list, labels, words

    def build_tensor(self, frames):
        temporalVolume = torch.FloatTensor(29, self.in_channels, 88, 88)
        if(self.augmentation):
            augmentations = transforms.Compose([
                StatefulRandomHorizontalFlip(0.5),
            ])
        else:
            augmentations = transforms.Compose([])

        if self.in_channels == 1:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((88, 88)),
                augmentations,
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.4161, ], [0.1688, ]),
            ])
        elif self.in_channels == 3:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((88, 88)),
                augmentations,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        for i in range(0, 29):
            frame = frames[i].permute(2, 0, 1)  # (C, H, W)
            temporalVolume[i] = transform(frame)

        temporalVolume = temporalVolume.transpose(1, 0)  # (C, D, H, W)
        return temporalVolume

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file = self.files[idx]
        video, _, _ = torchvision.io.read_video(self.video_paths[idx], pts_unit='sec')  # (Tensor[T, H, W, C])
        if self.estimate_pose:
            angle_frame = video[14].permute(2, 0, 1)
        else:
            angle_frame = 0
        frames = self.build_tensor(video)
        # print("WORD: ", self.words[label].lower())
        labels = MyLRWDataset.txt2arr(self.words[label].lower(), start=1)
        # print("LABELS: ", labels)

        frames_len = frames.shape[0]
        labels_len = len(labels)
        # if self.estimate_pose:
        # yaw = 0
        # else:
        #     yaw = self.poses[file]
        # print('SHAPE: ', frames.shape)
        # sample = {
        #     'frames': frames,
        #     'label': torch.LongTensor([label]),
        #     'word': self.words[label],
        #     'file': self.files[idx],
        #     'yaw': torch.FloatTensor([yaw]),
        #     'angle_frame': angle_frame,
        # }
        return torch.FloatTensor(frames), torch.tensor(labels), frames_len, labels_len
    

    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyLRWDataset.characters.index(c) + start)
        return np.array(arr)
        

    @staticmethod
    def arr2txt(arr, start):
        txt = []
        # ##print(arr.detach().cpu().numpy(), type(arr.detach().cpu().numpy()))
        # arr = [t[t != -1] for t in arr]
        arr = arr[arr != -1]
        # ##print(arr)
        for n in arr:
            if(int(n) >= start):
                txt.append(MyLRWDataset.characters[int(n) - start])     
        return ''.join(txt).strip()
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        # pre = -1
        pre = 0
        txt = []
        # print("ARRAY: ",arr)
        for n in arr:
            if(pre != n and n >= start):                
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.characters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyLRWDataset.characters[n - start])                
            pre = n
        
        return ''.join(txt).strip()
    
    @staticmethod
    def wer(predict, truth): 
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [editdistance.eval(p[0], p[1]) / len(p[1]) for p in word_pairs]
        return np.array(wer)

        
    @staticmethod
    def cer(predict, truth):        
        cer = [editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return np.array(cer)
    


if __name__ == "__main__":
    dataset = MyDataset(
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_",
        "/media/sadevans/T7 Shield/PERSONAL/Diplom/datasets/Vmeste/for_/labels/Vmeste_train_transcript_lengths_seg24s_0to100_5000units.csv",
        "train",
        "video"
    )


    # dataset.load_anno()
    # dataset
