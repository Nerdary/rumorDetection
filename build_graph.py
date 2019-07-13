#/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine

#数据集
datasets = ['semeval2017', 'semeval2019']

#读入数据集名
if len(sys.argv) != 2:
	sys.exit("Use: python train.py <dataset>")

#校验数据集是否存在
if sys.argv[1] not in datasets:
	sys.exit("wrong dataset name")

word_embeddings_dim = 300
word_vector_file = './data/glove.6B.300d.txt'
# word_vector_file = './data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
vocab, embd, word_vector_map = loadWord2Vec(word_vector_file)

#填入数据位置
doc_name_list = []
doc_train_list = []
doc_test_list = []
doc_content_list = []

#读入数据
doc = dict()
with open('./data/pdata/{}.rumor.dict'.format(sys.argv[1]), 'r', encoding='utf-8') as f:
    doc = eval(f.read())

f.close()

#填入数据
for key in doc:
    doc_name_list.append(key)
    doc[key]['id'] = key
    # print(doc[key])
    if doc[key]['source'] == 'train':
        doc_train_list.append(doc[key])
    elif doc[key]['source'] == 'test':
        doc_test_list.append(doc[key])

#编写训练样本序号
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name['id'])
    train_ids.append(train_id)

random.shuffle(train_ids)

#写入文件
train_ids_str = '\n'.join(str(index) for index in train_ids)
with open('./data/pdata/{}.train.index'.format(sys.argv[1]), 'w', encoding='utf-8') as f:
    f.write(train_ids_str)

f.close()

#写入测试样本序号
test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name['id'])
    test_ids.append(test_id)

random.shuffle(test_ids)

#写入文件
test_ids_str = '\n'.join(str(index) for index in test_ids)
with open('./data/pdata/{}.test.index'.format(sys.argv[1]), 'w', encoding='utf-8') as f:
    f.write(test_ids_str)

f.close()

#配对样本ID与内容
ids = train_ids + test_ids
shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append("{}\t{}\t{}".format(doc_name_list[int(id)], doc[doc_name_list[int(id)]]['source'], doc[doc_name_list[int(id)]]['label']))
    shuffle_doc_words_list.append(doc[doc_name_list[int(id)]]['text'])

shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

#写入文件
with open('./data/pdata/{}_shuffle.txt'.format(sys.argv[1]), 'w', encoding='utf-8') as f:
    f.write(shuffle_doc_name_str)

f.close()

with open('./data/pdata/{}_corpus_shuffle.txt'.format(sys.argv[1]), 'w', encoding='utf-8') as f:
    f.write(shuffle_doc_words_str)

f.close()

#建词表
word_freq = {}
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

#统计每个词出现的文档集合
word_doc_list = {}
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue

        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]

        appeared.add(word)

#统计每个词出现过的文档个数
word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

#将词转换为ID
word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

#写入文件
vocab_str = '\n'.join(vocab)
with open('./data/pdata/{}_vocab.txt'.format(sys.argv[1]), 'w', encoding='utf-8') as f:
    f.write(vocab_str)

f.close()

#准备标签文件
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])

label_list = list(label_set)

label_list_str = '\n'.join(label_list)
with open('./data/pdata/{}_labels.txt'.format(sys.argv[1]), 'w', encoding='utf-8') as f:
    f.write(label_list_str)

f.close()

#分割训练样本
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size

#存储
real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)
with open('./data/pdata/{}.real_train.name'.format(sys.argv[1]), 'w', encoding='utf-8') as f:
    f.write(real_train_doc_names_str)

f.close()

#
row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    if doc_len == 0:
        doc_len = 1

    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec += np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        data_x.append(doc_vec[j] / doc_len)

x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))

#将训练集标签转为one-hot
y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)

y = np.array(y)

#
test_size = len(test_ids)
row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    if doc_len == 0:
        doc_len = 1

    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        data_tx.append(doc_vec[j] / doc_len)

tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

#装载测试标签
ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)

ty = np.array(ty)

#初始化
word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

#为词典装载词向量
for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

#装载全部训练集
row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    if doc_len == 0:
        doc_len = 1
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        data_allx.append(doc_vec[j] / doc_len)

#将词典所有词加入
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))

row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

#所有词标签为0
for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)
#print(y.shape, ty.shape, allx.shape, ally.shape)

#
window_size = 10
windows = []

#将所有窗口取出
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

#统计窗口内词的词频（单窗口内算一次）
word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

#窗口内词对出现次数
word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

num_window = len(windows)

#计算窗口内词对的pmi
for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue

    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# for i in range(vocab_size):
#     for j in range(vocab_size):
#         if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
#             vector_i = np.array(word_vector_map[vocab[i]])
#             vector_j = np.array(word_vector_map[vocab[j]])
#             similarity = 1.0 - cosine(vector_i, vector_j)
#             #print(vocab[i], vocab[j], similarity)
#             if similarity > 0.9:
#                 row.append(train_size + i)
#                 col.append(train_size + j)
#                 weight.append(similarity)

#文档与词共现的次数
doc_word_freq = {}
for doc_id in range(len(shuffle_doc_words_list)):
    #print("doc_id", doc_id)
    doc_words = shuffle_doc_words_list[doc_id]
    #print("doc_words", shuffle_doc_name_list[doc_id])
    #print("doc_words", shuffle_doc_words_list[doc_id])
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

#加载树结构


node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

f = open("data/ind.{}.x".format(sys.argv[1]), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.y".format(sys.argv[1]), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.tx".format(sys.argv[1]), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.ty".format(sys.argv[1]), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.allx".format(sys.argv[1]), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.ally".format(sys.argv[1]), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.adj".format(sys.argv[1]), 'wb')
pkl.dump(adj, f)
f.close()
