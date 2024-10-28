import json
import numpy as np
from transformers import AutoTokenizer
from decimal import getcontext

from config import llm_model_name, llm_model_checkpoint, data_save_dir

getcontext().prec = 100

def cosine_similarity_vectors(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b) if norm_a and norm_b else 0
    return similarity

def add_laplace_noise_to_vector(vector, epsilon, delta_f_new):
    vector = np.asarray(vector, dtype=np.longdouble)
    beta_values = delta_f_new / (0.5 * epsilon) if epsilon > 0 else np.zeros_like(delta_f_new)
    noise = np.random.laplace(loc=0, scale=beta_values, size=len(beta_values))
    noisy_vector = vector + noise
    return noisy_vector

def perturb_prompt(prompt, epsilon, tokenizer, token_to_vector_dict, sorted_distance_data, delta_f_new):
    tokens = tokenizer.tokenize(prompt)
    delta_u = 1.0
    exp_factor = epsilon / (2 * delta_u)

    new_tokens = []
    for origin_token in tokens:
        origin_token = origin_token.lstrip()  # 去除前导空格
        origin_embed = token_to_vector_dict.get(origin_token) # 得到token的嵌入向量形式
        if origin_embed is None:
            new_tokens.append(origin_token)
            continue

        # 以下是噪声嵌入。得到加噪后的向量
        noise_embed = add_laplace_noise_to_vector(origin_embed, epsilon, delta_f_new)

        # 计算原始嵌入向量和加噪后的嵌入的向量的余弦相似度。
        similarity = cosine_similarity_vectors(origin_embed, noise_embed)

        # 取和原始token相似的单词
        sorted_similarity = sorted_distance_data.get(origin_token)
        if sorted_similarity is None:
            continue

        # 拿出来这个单词和相似度。
        token_only, similarity_only = sorted_similarity
        arr = np.flip(similarity_only)
        index = np.searchsorted(arr, similarity)
        index = len(arr) - index

        close_tokens = token_only[:index]
        close_similarities = similarity_only[:index]
        if not close_tokens:
            continue

        unnormalized_probabilities = np.exp(exp_factor * np.array(close_similarities))
        total_unnormalized_prob = np.sum(unnormalized_probabilities)
        probabilities = unnormalized_probabilities / total_unnormalized_prob
        selected_token = np.random.choice(close_tokens, p=probabilities)
        new_tokens.append(selected_token)

    token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    sentence = tokenizer.decode(token_ids)
    return sentence

def load_json_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return None

def perturb_prompts(prompt_list, epsilon_list, model_name, model_checkpoint, save_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

    token_2_embedding_dict = load_json_file(f"{save_dir}token_2_embedding_{model_name}.json")
    if token_2_embedding_dict is None:
        return

    token_sorted_distance_dict = load_json_file(f"{save_dir}token_sorted_similarity_dict_{model_name}.json")
    if token_sorted_distance_dict is None:
        return

    delta_f_new = load_json_file(f"{save_dir}sensitivity_of_embeddings_{model_name}.json")
    if delta_f_new is None:
        return
    delta_f_new = np.array(delta_f_new)

    result_list = []
    for prompt in prompt_list:
        # 输出原始提示
        print(f"Original prompt: {prompt}.")
        ptb_prompt_list = []
        for eps in epsilon_list:
            print("=" * 100)
            print(f"Add noise with eps: {eps} to original prompt.")
            #
            ptb_prompt = perturb_prompt(prompt, eps, tokenizer, token_2_embedding_dict, token_sorted_distance_dict, delta_f_new)
            print(f"====> perturbed prompt: {ptb_prompt}")

            ptb_prompt_list.append({"ptb_prompt": ptb_prompt, "eps": eps})

        result_list.append({"original_prompt": prompt, "perturbed_prompts": ptb_prompt_list})

    with open(f"{save_dir}perturb_result.json", 'w') as f:
        json.dump(result_list, f)
    print("perturb result saved")

if __name__ == "__main__":
    prompt = "Lisa, a 28-year-old woman, went to the hospital for a thorough examination after experiencing " \
             "unexplained weight loss, fatigue, and frequent infections. Tests showed abnormal blood cell counts " \
             "and compromised immune function, leading to a diagnosis of Idiopathic Immunodeficiency Syndrome. " \
             "Treatment options were discussed to boost her immune system."

    epsilon = [1, 2, 4, 6, 8]
    prompts = [prompt]
    perturb_prompts(prompts, epsilon, llm_model_name, llm_model_checkpoint, data_save_dir)
