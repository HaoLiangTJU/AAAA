import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import faiss
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def add_vector_to_index(embedding, index):
    vector = embedding.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    index.add(vector)

#你可以换成dinov2-base/large/giant模型
processor = AutoImageProcessor.from_pretrained('/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge_feature_extract/dinov2-small')
model = AutoModel.from_pretrained('/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge_feature_extract/dinov2-small').to(device)

data_folder = '/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge'
folders = os.listdir(data_folder)
images=[]
know_ref={"mzd":"毛泽东，中华人民共和国主席，共和国的缔造者，中国共产党的领导人",
"dengxiaoping":"邓小平，中华人民共和国主席，改革开放的总工程师，中国共产党的领导人",
"xijinping":"习近平，中华人民共和国主席，现任中国共产党的领导人",
"falungong":"法轮功，是一个邪教组织，有反党反政府的恶劣行径",
"jiangzemin":"江泽民，中华人民共和国主席，中国共产党的领导人",
"hujintao":"胡锦涛，中华人民共和国主席，中国共产党的领导人",
"guohui":"正常国徽",
"guoqi":"正常国旗",
"weiqi":"涉政的非法反动政府的伪政府旗帜",
"danghui":"正常党徽",
"zcjz":"正常军装",
"hushi":"正常胡适",
"zcsj":"正常书籍",
"zccj":"正常场景",
"zcrq":"正常人群",
"zcdt":"正常地图",
"hardcase":"正常内容",
"hateful":"有害涉政内容",
"zcms":"正常模式",
"xiong":"国家领导人-人形熊"}
i=0
know_based = {}
for folder in folders:
    print(folder)
    imgs = os.listdir(data_folder + '/' + folder)
    for img in imgs:
        images.append(data_folder+'/'+folder + '/' + img)
        know_based.update({i:[data_folder+'/'+folder + '/' + img,know_ref[folder]]})
        i = i + 1
with open("/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge_feature_extract/know_feature.json", 'w', encoding='utf-8') as f:
    json.dump(know_based, f, ensure_ascii=False, indent=4)
    print("加载入文件完成...")


# #feature dim 是384维，所以建立dim=384的index,type是FlatL2
index = faiss.IndexFlatL2(384)
#t0 = time.time()
for image_path in tqdm(images):
    img = Image.open(image_path).convert('RGB')
    with torch.no_grad():
        inputs = processor(images=img,return_tensors='pt').to(device)
        outputs = model(**inputs)
    features = outputs.last_hidden_state
    add_vector_to_index(features.mean(dim=1), index)

#print('Extraction done in: ', time.time() - t0)
faiss.write_index(index, '/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge_feature_extract/know.index')
