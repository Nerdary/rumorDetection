#!/usr/bin/python
#-*-coding:utf-8-*-
from nltk.corpus import stopwords
import nltk

import os
import json
import re

word_freq = dict()
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#过滤字符
def Filter(input):
    pattern1 = re.compile(r'http[a-zA-Z0-9.?/&=:]*')
    input = pattern1.sub("", input)
    # pattern2 = re.compile(r'[-,$()#+&*!?.":;/–：，。“”‘’=+]')
    # input = pattern2.sub(" ", input)
    r = ""
    words = input.strip().split()
    for word in words:
        word = word.lower()
        if '@' not in word:
            r += (word + ' ')
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    return r

#清理文本
def CleanText(rumor_c):
    for key in rumor_c:
        r = ""
        words = rumor_c[key]['text'].split(' ')
        for word in words:
            if word not in stop_words and word != '':
                if word_freq[word] > 1:
                    r += (word + ' ')

        rumor_c[key]['text'] = r
        return rumor_c

# #平衡数据
# def BalanceData(rumor_c):
#     labels = {}
#     for key in rumor_c:
#         #nkey = '{}-{}'.format(rumor_c[key]['source'], rumor_c[key]['label'])
#         if rumor_c[key]['label'] in labels:
#             labels[rumor_c[key]['label']] += 1
#         else:
#             labels[rumor_c[key]['label']] = 1
#
#     print(labels)
#     return rumor_c


#存储数据文件
def SaveData(rumor_l, extra_g, fn):
    fw = open("./data/pdata/{}.rumor.dict".format(fn), 'w', encoding='utf-8')
    fw.write(str(rumor_l))
    fw.close()
    fw = open("./data/pdata/{}.extra.g.dict".format(fn), 'w', encoding='utf-8')
    fw.write(str(extra_g))
    fw.close()

dataset_name = ['semeval2017']

for filename in dataset_name:
    if filename == 'semeval2017':
        #存转发关系、文本内容
        extra_graph = dict()
        rumor_content = dict()
        stance_label = dict()
        rumor_label = dict()
        train = 0
        test = 0
        source = 0

        #获取文本、传播结构(训练+测试)
        for sub_fn in os.listdir('./data/semeval2017/semeval2017-task8-dataset/rumoureval-data'):
            #print(sub_fn)
            for rumor_fn in os.listdir('./data/semeval2017/semeval2017-task8-dataset/rumoureval-data/{}'.format(sub_fn)):
                #获取源节点文本
                #print(rumor_fn)
                with open('./data/semeval2017/semeval2017-task8-dataset/rumoureval-data/{}/{}/source-tweet/{}.json'.format(sub_fn, rumor_fn, rumor_fn), 'r', encoding='utf-8') as load_f:
                    load_t = json.load(load_f)
                    source += 1
                    rumor_content[rumor_fn] = dict()
                    rumor_content[rumor_fn]['root'] = 1
                    #rumor_content[rumor_fn]['text'] = Filter(load_t['text'])
                    rumor_content[rumor_fn]['text'] = load_t['text']
                    if sub_fn == 'semeval2017-task8-test-data':
                        rumor_content[rumor_fn]['source'] = 'test'
                        test += 1
                    else:
                        rumor_content[rumor_fn]['source'] = 'train'
                        train += 1

                load_f.close()

                #获取转发节点文本
                replies_t_fn = os.listdir('./data/semeval2017/semeval2017-task8-dataset/rumoureval-data/{}/{}/replies'.format(sub_fn, rumor_fn))
                #print(replies_t_fn)
                for sub_replies_t_fn in replies_t_fn:
                    with open('./data/semeval2017/semeval2017-task8-dataset/rumoureval-data/{}/{}/replies/{}'.format(sub_fn, rumor_fn, sub_replies_t_fn), 'r', encoding='utf-8') as load_f:
                        load_t = json.load(load_f)
                        item = sub_replies_t_fn.replace('.json', '')
                        rumor_content[item] = dict()
                        rumor_content[item]['root'] = 0
                        #rumor_content[item]['text'] = Filter(load_t['text'])
                        rumor_content[item]['text'] = load_t['text']
                        if sub_fn == 'semeval2017-task8-test-data':
                            rumor_content[item]['source'] = 'test'
                            test += 1
                        else:
                            rumor_content[item]['source'] = 'train'
                            train += 1

                load_f.close()

                #获取转发关系
                with open('./data/semeval2017/semeval2017-task8-dataset/rumoureval-data/{}/{}/structure.json'.format(sub_fn, rumor_fn), 'r', encoding='utf-8') as load_f:
                    load_t = json.load(load_f)
                    extra_graph.update(load_t)

                load_f.close()

        #获取立场及谣言标签
        label_fn_list = os.listdir('./data/semeval2017/semeval2017-task8-dataset/traindev')
        for sub_fn in label_fn_list:
            if 'subtaskA' in sub_fn:
                with open('./data/semeval2017/semeval2017-task8-dataset/traindev/{}'.format(sub_fn), 'r', encoding='utf-8') as load_f:
                    load_t = json.load(load_f)
                    stance_label.update(load_t)

                load_f.close()
            elif 'subtaskB' in sub_fn:
                with open('./data/semeval2017/semeval2017-task8-dataset/traindev/{}'.format(sub_fn), 'r', encoding='utf-8') as load_f:
                    load_t = json.load(load_f)
                    rumor_label.update(load_t)

                load_f.close()

        for key in rumor_content:
            if key in stance_label:
                rumor_content[key]['label'] = stance_label[key]

            if key in rumor_label:
                rumor_content[key]['r_label'] = rumor_label[key]

        #SaveData(CleanText(rumor_content), extra_graph, "semeval2017")
        SaveData(rumor_content, extra_graph, "semeval2017")