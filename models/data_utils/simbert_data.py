import json

import torch
from torch.utils.data import Dataset,DataLoader
import time
from datetime import timedelta
import numpy as np

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def bulid_dataset(args):
    dataset = SimBertDataset(args.dataset_path)
    return dataset

class SimBertDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.text_list = []
        self.synonym_list = []
        i = 0
        with open(self.dataset_path,'r',encoding='utf-8') as fr:
            for line in fr:
                i += 1
                line = json.loads(line)
                text,synonyms = line['text'],line['synonyms']
                synonyms = [text] + synonyms
                np.random.shuffle(synonyms)
                text,synonym = synonyms[:2]
                self.text_list.append(text)
                self.text_list.append(synonym)
                self.synonym_list.append(synonym)
                self.synonym_list.append(text)
                if i%1000 == 0:
                    print('processing: {}'.format(i))

    def __getitem__(self, index):
        return self.text_list[index],self.synonym_list[index]

    def __len__(self):
        return len(self.text_list)




if __name__ == '__main__':

    mydataset = SimBertDataset('../../corpus/data_sample.json')
    testloader = DataLoader(mydataset, batch_size=4, shuffle=False)
    test_iter = iter(testloader)
    data = test_iter.next()
    print("test")
