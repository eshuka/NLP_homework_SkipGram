from scipy import spatial
import tensorflow as tf


def get_similar_word(query, final_embeddings, word2id):
    query_idx = word2id[query]
    query_vector = final_embeddings[query_idx]
    similar_list = []
    for word in word2id:
        db_idx = word2id[word]
        db_vector = final_embeddings[db_idx]
        similar_list.append([word, round(1 - spatial.distance.cosine(query_vector, db_vector), 5)])
    result = get_top_k_result(similar_list, 8)
    return result

def get_top_k_result(similar_list, k):
    result = (sorted(similar_list, key=lambda l: l[1], reverse=True))
    return result[1:k+1]


