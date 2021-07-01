# SimBert PyTorch版本
复现[ZhuiyiTechnology](https://github.com/ZhuiyiTechnology)提出的simbert，模型细节可以参考苏神的文章：https://kexue.fm/archives/7427

由于原simbert是基于tf1.14进行开发，debug不是很方便，而且原simbert为了精简模型参数在embedding层进行了低秩分解，导致模型转化成pytorch版本时无法转化的问题（如果有大佬知道如何转化的话还请赐教），作者基于simbert的思想基于pytorch和transformers进行复现。作者水平有限，如有错误，敬请各位大佬不吝赐教！

代码细节参考自：

@techreport{simbert,

title={SimBERT: Integrating Retrieval and Generation into BERT},

author={Jianlin Su},

year={2020},

url="https://github.com/ZhuiyiTechnology/simbert"

}

数据集来自：**ChineseSTS**

唐善成, 白云悦, 马付玉. 中文语义相似度训练集. 西安科技大学.2016. https://github.com/IAdmireu/ChineseSTS

Tang Shancheng, Bai Yunyue, Ma Fuyu. Chinese Semantic Text Similarity Trainning Dataset. Xi'an University of Science and Technology.2016. https://github.com/IAdmireu/ChineseSTS

