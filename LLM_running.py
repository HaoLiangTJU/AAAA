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

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = "Qwen/Qwen2.5-VL-3B-Instruct"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,  device_map="cuda"
)
k=0
with open("/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge_feature_extract/know_feature.json",'r') as load_f:
    know = json.load(load_f)

# default processer
processor = AutoProcessor.from_pretrained(model_dir,max_pixels=560*28*28)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
path_dir = "/data/lianghao/Huawei/相似图像集/大面积红色"
# path_dir = "/data/lianghao/Huawei/相似图像集/政府职员或政府职员穿着类、军装等正式服饰的人物"
# path_dir = "/data/lianghao/Huawei_add/pic/wujian"
# path_dir = "/data/lianghao/Huawei_add/pic/jianchu"
# path_dir = "/data/lianghao/Huawei_add/pic/waiyu"
img_dir = os.listdir(path_dir)
dino_processor = AutoImageProcessor.from_pretrained('/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge_feature_extract/dinov2-small')
dino_model = AutoModel.from_pretrained('/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge_feature_extract/dinov2-small').to(device)
index = faiss.read_index("/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge_feature_extract/know.index")
waste_sum=0
results = []
for img in img_dir:
    sz=0
    output_text = ''
    s = time.time()
    img_path = os.path.join(path_dir, img)
    knows=img_path
    image = Image.open(img_path)
    # Extract the features
    with torch.no_grad():
        inputs = dino_processor(images=image, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)
    # Normalize the features before search
    embeddings = outputs.last_hidden_state
    embeddings = embeddings.mean(dim=1)
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)

    d, i = index.search(vector, 1)

    if ("正常内容" == know[str(i[0][0])][1] and d[0][0] < 0.5):
        sz = 1
    elif("正常模式"==know[str(i[0][0])][1] and d[0][0]<0.2):
        sz=1
    elif("国家领导人-人形熊"==know[str(i[0][0])][1] and d[0][0]<0.3):
        sz=2
    elif("有害涉政内容"==know[str(i[0][0])][1] and d[0][0]<0.22):
        sz=2
    elif("正常国徽"==know[str(i[0][0])][1] and d[0][0]<0.3):
        sz=1
    elif("正常党徽"==know[str(i[0][0])][1] and d[0][0]<0.35):
        sz=1
    elif("正常军装"==know[str(i[0][0])][1] and d[0][0]<0.2):
        sz=1
    elif("正常胡适"==know[str(i[0][0])][1] and d[0][0]<0.25):
        sz=1
    elif("正常书籍"==know[str(i[0][0])][1] and d[0][0]<0.2):
        sz=1
    elif("正常国旗"==know[str(i[0][0])][1] and d[0][0]<0.2):
        sz=1
    elif ("正常场景" == know[str(i[0][0])][1] and d[0][0] < 0.3):
        sz = 1
    elif ("正常人群" == know[str(i[0][0])][1] and d[0][0] < 0.4):
        sz=1
    elif("正常地图" == know[str(i[0][0])][1] and d[0][0] < 0.15):
        sz=1
    elif(d[0][0]<0.4):
        know_sz = know[str(i[0][0])][1]
        prompt = """
            请分析该图像中的视觉和文本内容并结合视觉内容检索到的相关知识，请根据以下十类涉政隐喻评判标准，判断该图像是否具有涉及中国政治的隐喻内容，并输出涉政或不涉政，仅输出“涉政”或“不涉政”：
        视觉内容检索到的相关知识：{}
        涉政隐喻评判标准：1）领导人：只要出现领导人形象就算涉政隐喻。
        2）国家分裂势力：出现和台独、疆独、藏独、港独有关内容
        3）中国地图：对地图进行恶意篡改
        4）历史敏感事件：民运、法轮功、文革
        5）侮辱国旗：损坏中国国旗的漫画或燃烧中国国旗
        6）国际关系：对中国国际关系隐喻
        7）伪旗：出现国民党、满洲国、清政府、法西斯等旗帜
        8）政治制度：讽刺中国政治制定（例如宣传中国官方统计数据造假）
        9）朝鲜负面漫画
        10）辱华：侮辱华人和国家形象
                """.format(know_sz)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        print('---------------------')
        print(img)
        prompt = """
                    请你分析输入图像的视觉与可见文本内容，并按下述定义分别给出 A、B、C 的答案。
                    最终只输出一行，格式必须为： 是/否,是/否,涉政/不涉政
                    使用英文逗号分隔；
                    不允许输出引号、标签、空格、换行、列表、解释或任何额外字符；
                    若不确定，一律判为“否”。
                    你的输出必须匹配正则：^(是|否),(是|否),(涉政|不涉政)$
                    A.含熊特定场景检测：图像中是否存在有熊的特定场景,存在→“是”，否则→“否”。
                        特定场景：1）拟人化熊与政客握手;
                        2）熊与坦克车、推土机、挖掘机;
                        3）熊与天安门;
                        4）熊与阅兵;
                        5）熊+普京形象/金正恩形象/国旗/国徽/武器/中国形象/习近平/党旗/党徽/人民大会堂
                        6）维尼熊/黄色熊。
                    B.合规视觉内容检测：图像是否存在以下四个视觉内容中的某一个,如果存在，请回答“是”；反之，请回答“否”。
                        视觉内容：1）室内会议现场且不包含习近平、胡锦涛等中国领导人;
                        2）幻灯片、书法作品或语文卷子等多文字内容;
                        3）红头文件或盖有印章的文件;
                        4）正常红色场景;
                    C.涉政隐喻检测：请根据以下判断标准，判断图片是否构成对中国政治的 “仇恨性讽刺”,并输出涉政或不涉政。
                        判断标准：“仇恨性讽刺” 需同时满足：1. 使用象征、夸张、影射等讽刺手法；2. 存在煽动敌意、非人化、鼓吹极端蔑视等仇恨意图。若仅为一般性政治批评、非仇恨性政治漫画、历史 / 艺术表达、事实陈述或新闻报道，均不属于 “仇恨性讽刺”。
                        输出要求：仅需基于上述标准判断，明确满足 “仇恨性讽刺” 全部要素时，输出 “涉政”；存在任何疑问、模糊不清或属于排除情况时，输出 “不涉政”。
                """
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        with torch.no_grad():
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,padding=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=7)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )   
        print('output后: {}'.format(output_text))
        print('output[0] : {}'.format(output_text[0]))
        step1, step2, step3 = output_text[0].split(",")
        step1_map = {'是':2}
        step2_map = {'是':1}
        step3_map = {'涉政':2, '不涉政':1}

        sz = step1_map.get(step1) or step2_map.get(step2) or step3_map.get(step3)
        
        print(sz)
        if(step1 == '是'):
            sz=2
        elif(step2 == '是'):
            sz=1
        elif(step3 == '涉政'):
            sz=2
        elif(step3 == '不涉政'):
            sz=1
    if(sz==1):
        output_text=["不涉政"]
    elif(sz==2):
        output_text=["涉政"]
    else:
        with torch.no_grad():
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,padding=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=3)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )   
        print('output后 : {}'.format(output_text))
        print('output[0] : {}'.format(output_text[0]))
    e = time.time()
    if("涉政" in output_text[0] and "不涉政" not in output_text[0]):
        print(img_path,output_text)
        results.append('------------------------------------------------------\n' + img_path + '   ' + output_text[0] + '\n' +
                      'output : {}\n'.format(output_text) + 'output[0] : {}\n'.format(output_text[0]))
        k=k+1
    waste_sum+=(e-s)

for i in range(len(results)):
    print(results[i])
print("avg_waste_time: ", waste_sum/len(img_dir))
print("涉政样本：",k)
print("总样本数：", len(img_dir))