import json
import re

import numpy as np
import tqdm
# from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import llm_model_checkpoint, llm_model_name, data_save_dir
from utils import NumpyEncoder


def contains_english_chars(string):
    pattern = r'[a-zA-Z]'
    match = re.search(pattern, string)
    return bool(match)


def contains_non_english_chars(string):
    pattern = r'[^a-zA-Z]'
    match = re.search(pattern, string)
    return bool(match)


# def filter_tokens(token2index):
#     filtered_index2token = {}
#     for key, idx in tqdm.tqdm(token2index.items()):
#         #     print(val)
#         if key.startswith('<'):
#             continue
#
#         if not key.startswith('Ġ'):
#             continue
#
#         val_ = key.replace("Ġ", "")
#
#         if val_ == val_.upper():
#             continue
#
#         if contains_non_english_chars(val_):
#             continue
#
#         if 3 < len(val_) < 16 and contains_english_chars(val_):
#             filtered_index2token[idx] = key
#     #         if len(filtered_index2token) > 10000:
#     #             break
#     return filtered_index2token
def filter_tokens(token2index):
    filtered_index2token = {}
    total = len(token2index)
    counts = {
        "special_token": 0,
        "all_upper": 0,
        "non_english": 0,
        "length": 0,
        "kept": 0
    }

    for token, idx in tqdm.tqdm(token2index.items()):
        # BERT的特殊token通常以[开头和]结尾，如[CLS], [SEP], [MASK]
        if token.startswith('[') or token.startswith('##'):
            counts["special_token"] += 1
            continue

        # WordPiece tokenizer会在子词前加##，我们要去掉它
        if '##' in token:
            val_ = token.replace('##', '')
        else:
            val_ = token

        # 跳过全大写的token
        if val_ == val_.upper() and len(val_) > 1:  # 允许单个大写字母
            counts["all_upper"] += 1
            continue

        # 检查是否包含非英文字符
        if contains_non_english_chars(val_):
            counts["non_english"] += 1
            continue

        # 检查长度 (可以根据需要调整长度限制)
        if not (1 < len(val_) < 16 and contains_english_chars(val_)):
            counts["length"] += 1
            continue

        filtered_index2token[idx] = token
        counts["kept"] += 1

    print(f"\n总token数: {total}")
    print("Token统计:")
    for reason, count in counts.items():
        print(f"{reason}: {count}")

    # 打印一些保留的token示例
    print("\n保留的token示例:")
    for idx, token in list(filtered_index2token.items())[:10]:
        print(f"idx: {idx}, token: {token}")

    return filtered_index2token


def cosine_similarity(embedding_matrix1, embedding_matrix2):
    # Compute dot product of the matrices
    dot_product = np.dot(embedding_matrix1, embedding_matrix2.T)

    # Compute the norm of the matrices
    norm_matrix1 = np.linalg.norm(embedding_matrix1, axis=1)
    norm_matrix2 = np.linalg.norm(embedding_matrix2, axis=1)

    # Compute cosine similarity
    similarity = dot_product / (np.outer(norm_matrix1, norm_matrix2))

    return similarity

# 每个token创建一个按相似度排序的token列表
def create_sorted_similarities_for_tokens(token_list, similarity_matrix):
    # 存储结果的字典
    tokens_with_sorted_similarity = dict()
    # 单词数组
    token_array = np.array(token_list)
    for idx, token in tqdm.tqdm(enumerate(token_list)):
        # # 获取当前token与所有其他token的相似度
        similarity_array = similarity_matrix[idx]
        # 获取排序后的索引（降序）
        sorted_indices = np.argsort(similarity_array)[::-1]
        # 按相似度排序的tokens  对应的相似度值
        tokens_with_sorted_similarity[token] = [token_array[sorted_indices], similarity_array[sorted_indices]]
    return tokens_with_sorted_similarity


def create_sensitivity_of_embeddings(all_embedding_matrix):
    n_dimensions = all_embedding_matrix.shape[1]
    delta_f_new = np.zeros(n_dimensions)
    for dim in tqdm.trange(n_dimensions):
        dim_data = all_embedding_matrix[:, dim]
        sorted_dim_data = np.sort(dim_data)
        differences = sorted_dim_data[-1] - sorted_dim_data[0]
        delta_f_new[dim] = differences
    return delta_f_new


def get_embedding(model):
    # 获取嵌入层权重
    embedding_weights = model.get_input_embeddings().weight
    # 把嵌入层权重转换为numpy数组。
    return embedding_weights.detach().numpy()


def compute_token_2_embedding(index_2_token_dict, embedding_weights, model_name, save_dir):
    print(f"开始处理 {len(index_2_token_dict)} 个tokens...")

    # 预分配字典大小
    token_2_embedding_dict = {}

    # 批处理
    batch_size = 1000
    tokens = list(index_2_token_dict.items())

    for i in tqdm.tqdm(range(0, len(tokens), batch_size)):
        batch = tokens[i:i + batch_size]
        for idx, token in batch:
            # 直接转换为Python列表，避免numpy类型
            token_2_embedding_dict[token] = [float(x) for x in embedding_weights[idx]]

    # 分批写入文件
    file_full_path = save_dir + f'token_2_embedding_{model_name}.json'
    print(f"正在保存到 {file_full_path}...")

    with open(file_full_path, 'w') as f:
        # 使用更高效的json dumps
        json.dump(token_2_embedding_dict, f, ensure_ascii=False)

    print("保存完成!")
    return token_2_embedding_dict

def compute_embedding_similarity_matrix(token_2_embedding, model_name, save_dir):
    # 将嵌入向量字典形式转为数组形式。如果有N个token，每个embedding维度是M，则得到一个形状为(N, M)的矩阵
    embedding_matrix = np.array(list(token_2_embedding.values()))

    # 计算所有embedding向量之间的余弦相似度。
    # 结果是一个NxN的矩阵，其中第(i,j)个元素表示第i个token和第j个token的嵌入向量的余弦相似度。
    # 余弦相似度的范围是[-1, 1]，1表示方向完全相同，-1表示方向完全相反，0表示正交
    similarity_matrix = cosine_similarity(embedding_matrix, embedding_matrix)
    file_full_path = save_dir + "similarity_matrix_{}.npy".format(model_name)
    np.save(file_full_path, similarity_matrix, allow_pickle=True)
    print("similarity_matrix shape:", similarity_matrix.shape)
    print(f"Saved similarity_matrix to {file_full_path}.")
    return similarity_matrix


def compute_token_2_sorted_similarity(token_list, similarity_matrix, model_name, save_dir):
    _token_sorted_similarity_dict = create_sorted_similarities_for_tokens(token_list, similarity_matrix)
    file_full_path = save_dir + 'token_sorted_similarity_dict_{}.json'.format(model_name)
    with open(file_full_path, 'w') as f:
        json.dump(_token_sorted_similarity_dict, f, cls=NumpyEncoder)
    print(f"Saved token_sorted_similarity_dict to {file_full_path}.")
    return _token_sorted_similarity_dict


def compute_embedding_sensitivity(embedding_weights, model_name, save_dir):
    _delta_f_new = create_sensitivity_of_embeddings(embedding_weights)
    file_full_path = save_dir + 'sensitivity_of_embeddings_{}.json'.format(model_name)
    with open(file_full_path, 'w') as f:
        json.dump(_delta_f_new, f, cls=NumpyEncoder)
    print(f"Saved delta_f_new! to {file_full_path}.")
    return _delta_f_new


def create_perturbation_files(llm_model_name, llm_model_checkpoint, save_dir):

    # 加载预训练的分词器和语言模型
    tokenizer = AutoTokenizer.from_pretrained(llm_model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(llm_model_checkpoint)

    # 获取模型的词汇表，将词映射到索引 30522的字典
    token2index = tokenizer.get_vocab()

    # 模型中提取嵌入权重并存储为 NumPy 数组（30522*768）
    embedding_weights_np = get_embedding(model)

    # 这个函数的主要作用是从输入的词标记中筛选出符合特定条件的标记，以便后续处理。
    # 这些条件包括标记的格式、长度、以及是否包含英语字符。最终，函数返回一个经过严格筛选的词标记索引字典。
    # TODO 这里需要根据tokenizer的词汇表修改合适的函数
    filtered_index2token = filter_tokens(token2index)

    token_2_embedding = compute_token_2_embedding(filtered_index2token, embedding_weights_np, llm_model_name, save_dir)
    # 得到两个词向量矩阵的相似度
    similarity_matrix = compute_embedding_similarity_matrix(token_2_embedding, llm_model_name, save_dir)

    # 存的是单词
    token_list = list(token_2_embedding.keys())

    # 每个token创建一个按相似度排序的token列表
    compute_token_2_sorted_similarity(token_list, similarity_matrix, llm_model_name, save_dir)

    compute_embedding_sensitivity(embedding_weights_np, llm_model_name, save_dir)

    print("Perturbation Files Creation Is Completed!")


if __name__ == "__main__":
    create_perturbation_files(llm_model_name, llm_model_checkpoint, data_save_dir)

# 需要运行完成才能运行下面的程序。不然会报错。
