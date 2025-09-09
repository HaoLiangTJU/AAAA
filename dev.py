from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,AutoImageProcessor, AutoModel
from qwen_vl_utils import process_vision_info
# from modelscope import snapshot_download
import time
import os
import torch
from PIL import Image
import faiss
import numpy as np
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dino_processor = AutoImageProcessor.from_pretrained('/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge_feature_extract/dinov2-small')
dino_model = AutoModel.from_pretrained('/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge_feature_extract/dinov2-small').to(device)
index = faiss.read_index("/data/lianghao/Huawei/merged.index")


img_path = '/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge/hujintao/0.jpg'
image = Image.open(img_path)
with torch.no_grad():
    inputs = dino_processor(images=image, return_tensors="pt").to(device)
    outputs = dino_model(**inputs)
embeddings = outputs.last_hidden_state
embeddings = embeddings.mean(dim=1)
vector = embeddings.detach().cpu().numpy()
vector = np.float32(vector)
faiss.normalize_L2(vector)

d, i = index.search(vector, 1)

print(d, i)