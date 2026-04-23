import json
import random
import os

# ==========================================
# 核心配置区
# ==========================================
# 输入文件
ADE_TRAIN_FILE = "ade_unified_train.jsonl"
ADE_VAL_FILE = "ade_unified_validation.jsonl"
ADE_TEST_FILE = "ade_unified_test.jsonl"

DDI_TRAIN_FILE = "ddi_unified_train.jsonl"
DDI_TEST_FILE = "ddi_unified_test.jsonl"

# 输出文件
TRAIN_OUTPUT_FILE = "merged_chatml_train.jsonl"
VAL_OUTPUT_FILE = "mixed_chatml_validation.jsonl"
TEST_OUTPUT_FILE = "merged_chatml_test.jsonl"

# 让模型“双修”的超级系统提示词
SYSTEM_PROMPT = """You are an expert medical information extraction system. Your task is to read the provided medical text and extract all Adverse Drug Event (ADE) and Drug-Drug Interaction (DDI) relations.

Please strictly classify the relations according to the following 5 definitions:
1. "ADE" (Adverse Drug Event): A drug causes a specific disease, symptom, or adverse physiological reaction.
2. "DDI-mechanism": Describes the physical/chemical mechanism of interaction between two drugs (e.g., changes in absorption, metabolism, excretion, or serum concentration).
3. "DDI-effect": Describes the specific clinical consequences or altered pharmacodynamics when two drugs are co-administered (e.g., increased analgesia, increased toxicity).
4. "DDI-advice": Specific clinical guidance or warnings given by doctors or labels (e.g., not recommended to co-administer, dosage adjustment required).
5. "DDI-int" (Unspecified interaction): The text merely mentions that an interaction exists between two drugs without specifying the mechanism, effect, or advice.

You must strictly output a JSON list. Each element in the list must contain "head_entity", "tail_entity", and "relation_type" (which must be exactly one of the 5 labels above). If no relevant relations are found in the text, output an empty list []."""

def load_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到文件: {file_path}")
        return data
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
            
    # 【检查点】帮你打印第一行看看格式对不对
    if len(data) > 0:
        print(f"\n👀 检查 {file_path} 的第一条数据:")
        print(json.dumps(data[0], ensure_ascii=False, indent=2))
        
    return data

def convert_to_chatml(dataset):
    chatml_dataset = []
    for item in dataset:
        text = item['text']
        relations = item['relations']
        message_format = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
                {"role": "assistant", "content": json.dumps(relations, ensure_ascii=False)}
            ]
        }
        chatml_dataset.append(message_format)
    return chatml_dataset

def save_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# ==========================================
# 第一步：物理读取数据
# ==========================================
print("⏳ 正在读取数据...")
ade_train = load_jsonl(ADE_TRAIN_FILE)
ade_val = load_jsonl(ADE_VAL_FILE)
ade_test = load_jsonl(ADE_TEST_FILE)

ddi_train_full = load_jsonl(DDI_TRAIN_FILE)
ddi_test = load_jsonl(DDI_TEST_FILE)

# ==========================================
# 第二步：数据切分 (DDI Train -> 90% Train, 10% Val)
# ==========================================
# 为了保证可重复性，先打乱 DDI Train 数据
random.seed(42)
random.shuffle(ddi_train_full)

split_idx = int(0.9 * len(ddi_train_full))
ddi_train_90 = ddi_train_full[:split_idx]
ddi_train_10 = ddi_train_full[split_idx:]

print(f"\n👉 DDI Train 分割完成: 90% ({len(ddi_train_90)} 条), 10% ({len(ddi_train_10)} 条)")

# ==========================================
# 第三步：各子集物理合并并打乱
# ==========================================
# 1. Merged Train: ADE train + 90% DDI train -> Global Shuffle
merged_train_raw = ade_train + ddi_train_90
random.shuffle(merged_train_raw)
print(f"✅ 训练集构建完成 (ADE_train + 90% DDI_train)，共 {len(merged_train_raw)} 条。已进行全局打乱。")

# 2. Mixed Validation: ADE validation + 10% DDI train
mixed_val_raw = ade_val + ddi_train_10
random.shuffle(mixed_val_raw)
print(f"✅ 验证集构建完成 (ADE_val + 10% DDI_train)，共 {len(mixed_val_raw)} 条。")

# 3. Merged Test: ADE test + DDI test
merged_test_raw = ade_test + ddi_test
print(f"✅ 测试集构建完成 (ADE_test + DDI_test)，共 {len(merged_test_raw)} 条。绝对未参与训练。")

# ==========================================
# 第四步：转换为 ChatML 格式并保存
# ==========================================
print("\n⏳ 正在转换为 ChatML 格式并保存文件...")
chatml_train = convert_to_chatml(merged_train_raw)
chatml_val = convert_to_chatml(mixed_val_raw)
chatml_test = convert_to_chatml(merged_test_raw)

save_jsonl(chatml_train, TRAIN_OUTPUT_FILE)
save_jsonl(chatml_val, VAL_OUTPUT_FILE)
save_jsonl(chatml_test, TEST_OUTPUT_FILE)

print(f"\n🎉 所有数据合并与 ChatML 格式化彻底完成！")
print(f" - 训练文件已保存至: {TRAIN_OUTPUT_FILE}")
print(f" - 验证文件已保存至: {VAL_OUTPUT_FILE}")
print(f" - 测试文件已保存至: {TEST_OUTPUT_FILE}")