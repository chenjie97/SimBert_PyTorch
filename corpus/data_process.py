import json

import pandas as pd


def to_simbert_data():
    """构造一批假数据，转化成simbert可以接收的形式"""
    df = pd.read_table('./ChineseSTS-master/simtrain_to05sts.txt',names=['idx','text','idx2','synonyms','sim'])
    print(df.head())
    fw = open('data_similarity.json','w',encoding='utf-8')
    for i,row in df.iterrows():
        text = row['text']
        synonym = row['synonyms']
        sim = row['sim']
        if sim < 4:
            continue
        json.dump({'text':text,'synonyms':[synonym]},fw,ensure_ascii=False)
        fw.write('\n')
    fw.close()



if __name__ == '__main__':
    to_simbert_data()