import faiss
import json

def merge_faiss_and_json(index_file1, json_file1, index_file2, json_file2, out_index, out_json):
    # 1. 加载 index
    index1 = faiss.read_index(index_file1)
    index2 = faiss.read_index(index_file2)

    # 2. 把 index2 的向量取出来，再 add 到 index1
    xb = index2.reconstruct_n(0, index2.ntotal)  # 取出所有向量
    index1.add(xb)  # 加到 index1

    # 保存合并后的 index
    faiss.write_index(index1, out_index)

    # 3. 加载 json
    with open(json_file1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(json_file2, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    # 4. 计算偏移量
    offset = len(data1)

    # 5. 合并 json（第二组 key 要加偏移量）
    merged_json = {}
    for k, v in data1.items():
        merged_json[str(k)] = v
    for k, v in data2.items():
        merged_json[str(int(k) + offset)] = v

    # 6. 保存合并后的 json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged_json, f, ensure_ascii=False, indent=4)


# 用法示例
merge_faiss_and_json(
    "/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge_feature_extract/know.index", "/data/lianghao/Huawei/huawei_meme_update12/reasoning/knowledge_feature_extract/know_feature.json",
    "/data/lianghao/Huawei/know.index", "/data/lianghao/Huawei/know_feature.json",
    "/data/lianghao/Huawei/merged.index", "/data/lianghao/Huawei/merged.json"
)
