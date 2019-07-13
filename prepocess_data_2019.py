#!/usr/bin/python
#-*-coding:utf-8-*-
from nltk.corpus import stopwords
from tokenizer import tokenizer
import nltk

import os
import json
import re

word_freq = dict()
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
T = tokenizer.TweetTokenizer()

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

dataset_name = ['semeval2019']

for filename in dataset_name:
    if filename == 'semeval2019':
        #存转发关系、文本内容
        extra_graph = dict()
        rumor_content = dict()
        stance_label = dict()
        rumor_label = dict()
        train = 0
        test = 0
        source = 0

        #获取文本、传播结构(训练+测试)

        #获取训练集
        # twitter
        for sub_fn in os.listdir('./data/RumourEval2019/rumoureval-2019-training-data/twitter-english'):
        # print(sub_fn)
            for rumor_fn in os.listdir('./data/RumourEval2019/rumoureval-2019-training-data/twitter-english/{}'.format(sub_fn)):
            #获取源节点文本
            # print(rumor_fn)
                with open('./data/RumourEval2019/rumoureval-2019-training-data/twitter-english/{}/{}/source-tweet/{}.json'.format(sub_fn, rumor_fn, rumor_fn), 'r', encoding='utf-8') as load_f:
                    load_t = json.load(load_f)
                    source += 1
                    rumor_content[rumor_fn] = dict()
                    rumor_content[rumor_fn]['root'] = 1
                    rumor_content[rumor_fn]['text'] = ' '.join( T.tokenize(load_t['text']) )

                    rumor_content[rumor_fn]['source'] = 'train'
                    train += 1
                load_f.close()

                #获取转发节点文本
                replies_t_fn = os.listdir('./data/RumourEval2019/rumoureval-2019-training-data/twitter-english/{}/{}/replies'.format(sub_fn, rumor_fn))
                #print(replies_t_fn)
                for sub_replies_t_fn in replies_t_fn:
                    with open('./data/RumourEval2019/rumoureval-2019-training-data/twitter-english/{}/{}/replies/{}'.format(sub_fn, rumor_fn, sub_replies_t_fn), 'r', encoding='utf-8') as load_f:
                        load_t = json.load(load_f)
                        item = sub_replies_t_fn.replace('.json', '')
                        rumor_content[item] = dict()
                        rumor_content[item]['root'] = 0
                        rumor_content[item]['text'] = ' '.join( T.tokenize(load_t['text']) )

                        rumor_content[item]['source'] = 'train'
                        train += 1
                    load_f.close()

                #获取转发关系
                with open('./data/RumourEval2019/rumoureval-2019-training-data/twitter-english/{}/{}/structure.json'.format(sub_fn, rumor_fn), 'r', encoding='utf-8') as load_f:
                    load_t = json.load(load_f)
                    extra_graph.update(load_t)
                load_f.close()
        
        # reddit 训练集
        for rumor_fn in os.listdir('./data/RumourEval2019/rumoureval-2019-training-data/reddit-training-data'):
        #获取源节点文本
        # print(rumor_fn)
            with open('./data/RumourEval2019/rumoureval-2019-training-data/reddit-training-data/{}/source-tweet/{}.json'.format(rumor_fn, rumor_fn), 'r', encoding='utf-8') as load_f:
                load_t = json.load(load_f)
                source += 1
                rumor_content[rumor_fn] = dict()
                rumor_content[rumor_fn]['root'] = 1
                rumor_content[rumor_fn]['text'] = load_t['data']['children'][0]['data']['selftext']

                rumor_content[rumor_fn]['source'] = 'train'
                train += 1
            load_f.close()

            #获取转发节点文本
            replies_t_fn = os.listdir('./data/RumourEval2019/rumoureval-2019-training-data/reddit-training-data/{}/replies'.format(rumor_fn))
            #print(replies_t_fn)
            for sub_replies_t_fn in replies_t_fn:
                with open('./data/RumourEval2019/rumoureval-2019-training-data/reddit-training-data/{}/replies/{}'.format(rumor_fn, sub_replies_t_fn), 'r', encoding='utf-8') as load_f:
                    load_t = json.load(load_f)

                    if (load_t['kind']=='more'):
                        load_f.close()
                        continue

                    item = sub_replies_t_fn.replace('.json', '')
                    rumor_content[item] = dict()
                    rumor_content[item]['root'] = 0
                    # print(load_t)
                    rumor_content[item]['text'] = load_t['data']['body']

                    rumor_content[item]['source'] = 'train'
                    train += 1
                load_f.close()

            #获取转发关系
            with open('./data/RumourEval2019/rumoureval-2019-training-data/reddit-training-data/{}/structure.json'.format(rumor_fn), 'r', encoding='utf-8') as load_f:
                load_t = json.load(load_f)
                extra_graph.update(load_t)
            load_f.close()
        
        # reddit 验证集
        for rumor_fn in os.listdir('./data/RumourEval2019/rumoureval-2019-training-data/reddit-dev-data'):
        #获取源节点文本
        # print(rumor_fn)
            with open('./data/RumourEval2019/rumoureval-2019-training-data/reddit-dev-data/{}/source-tweet/{}.json'.format(rumor_fn, rumor_fn), 'r', encoding='utf-8') as load_f:
                load_t = json.load(load_f)
                source += 1
                rumor_content[rumor_fn] = dict()
                rumor_content[rumor_fn]['root'] = 1
                rumor_content[rumor_fn]['text'] = load_t['data']['children'][0]['data']['selftext']

                rumor_content[rumor_fn]['source'] = 'train'
                train += 1
            load_f.close()

            #获取转发节点文本
            replies_t_fn = os.listdir('./data/RumourEval2019/rumoureval-2019-training-data/reddit-dev-data/{}/replies'.format(rumor_fn))
            #print(replies_t_fn)
            for sub_replies_t_fn in replies_t_fn:
                with open('./data/RumourEval2019/rumoureval-2019-training-data/reddit-dev-data/{}/replies/{}'.format(rumor_fn, sub_replies_t_fn), 'r', encoding='utf-8') as load_f:
                    load_t = json.load(load_f)

                    if (load_t['kind']=='more'):
                        load_f.close()
                        continue

                    item = sub_replies_t_fn.replace('.json', '')
                    rumor_content[item] = dict()
                    rumor_content[item]['root'] = 0
                    rumor_content[item]['text'] = load_t['data']['body']

                    rumor_content[item]['source'] = 'train'
                    train += 1
                load_f.close()

            #获取转发关系
            with open('./data/RumourEval2019/rumoureval-2019-training-data/reddit-dev-data/{}/structure.json'.format(rumor_fn), 'r', encoding='utf-8') as load_f:
                load_t = json.load(load_f)
                extra_graph.update(load_t)
            load_f.close()

        #获取测试集

        # twitter
        for sub_fn in os.listdir('./data/RumourEval2019/rumoureval-2019-test-data/twitter-en-test-data'):
        # print(sub_fn)
            for rumor_fn in os.listdir('./data/RumourEval2019/rumoureval-2019-test-data/twitter-en-test-data/{}'.format(sub_fn)):
            #获取源节点文本
            # print(rumor_fn)
                #if (rumor_fn == '852844529778601985'):
                #    print("At least in 852844529778601985 case")

                with open('./data/RumourEval2019/rumoureval-2019-test-data/twitter-en-test-data/{}/{}/source-tweet/{}.json'.format(sub_fn, rumor_fn, rumor_fn), 'r', encoding='utf-8') as load_f:
                    load_t = json.load(load_f)
                    source += 1
                    rumor_content[rumor_fn] = dict()
                    rumor_content[rumor_fn]['root'] = 1
                    rumor_content[rumor_fn]['text'] = ' '.join( T.tokenize(load_t['text']) )
                    rumor_content[rumor_fn]['source'] = 'test'
                    test += 1
                load_f.close()

                #获取转发节点文本
                replies_t_fn = os.listdir('./data/RumourEval2019/rumoureval-2019-test-data/twitter-en-test-data/{}/{}/replies'.format(sub_fn, rumor_fn))
                #print(replies_t_fn)
                for sub_replies_t_fn in replies_t_fn:
                    with open('./data/RumourEval2019/rumoureval-2019-test-data/twitter-en-test-data/{}/{}/replies/{}'.format(sub_fn, rumor_fn, sub_replies_t_fn), 'r', encoding='utf-8') as load_f:
                        load_t = json.load(load_f)
                        item = sub_replies_t_fn.replace('.json', '')
                        rumor_content[item] = dict()
                        rumor_content[item]['root'] = 0
                        rumor_content[item]['text'] = ' '.join( T.tokenize(load_t['text']) )
                        rumor_content[item]['source'] = 'test'
                        test += 1
                    load_f.close()

                #获取转发关系
                with open('./data/RumourEval2019/rumoureval-2019-test-data/twitter-en-test-data/{}/{}/structure.json'.format(sub_fn, rumor_fn), 'r', encoding='utf-8') as load_f:
                    load_t = json.load(load_f)
                    extra_graph.update(load_t)
                load_f.close()

        # reddit
        for rumor_fn in os.listdir('./data/RumourEval2019/rumoureval-2019-test-data/reddit-test-data'):
        #获取源节点文本
        # print(rumor_fn)
            with open('./data/RumourEval2019/rumoureval-2019-test-data/reddit-test-data/{}/source-tweet/{}.json'.format(rumor_fn, rumor_fn), 'r', encoding='utf-8') as load_f:
                load_t = json.load(load_f)
                source += 1
                rumor_content[rumor_fn] = dict()
                rumor_content[rumor_fn]['root'] = 1
                rumor_content[rumor_fn]['text'] = load_t['data']['children'][0]['data']['selftext']
                rumor_content[rumor_fn]['source'] = 'test'
                test += 1
            load_f.close()

            #获取转发节点文本
            replies_t_fn = os.listdir('./data/RumourEval2019/rumoureval-2019-test-data/reddit-test-data/{}/replies'.format(rumor_fn))
            #print(replies_t_fn)
            for sub_replies_t_fn in replies_t_fn:
                with open('./data/RumourEval2019/rumoureval-2019-test-data/reddit-test-data/{}/replies/{}'.format(rumor_fn, sub_replies_t_fn), 'r', encoding='utf-8') as load_f:
                    load_t = json.load(load_f)

                    if (load_t['kind']=='more'):
                        load_f.close()
                        continue

                    item = sub_replies_t_fn.replace('.json', '')
                    rumor_content[item] = dict()
                    rumor_content[item]['root'] = 0
                    rumor_content[item]['text'] = load_t['data']['body']
                    rumor_content[item]['source'] = 'test'
                    test += 1
                load_f.close()

            #获取转发关系
            with open('./data/RumourEval2019/rumoureval-2019-test-data/reddit-test-data/{}/structure.json'.format(rumor_fn), 'r', encoding='utf-8') as load_f:
                load_t = json.load(load_f)
                extra_graph.update(load_t)
            load_f.close()


        #获取立场及谣言标签

        # 测试集标签
        with open('./data/RumourEval2019/final_eval_key.json', 'r', encoding='utf-8') as load_f:
            load_t = json.load(load_f)
            stance_label.update(load_t['subtaskaenglish'])
            rumor_label.update(load_t['subtaskbenglish'])
        load_f.close()
           
        # 训练集标签
        with open('./data/RumourEval2019/rumoureval-2019-training-data/train-key.json', 'r', encoding='utf-8') as load_f:
            load_t = json.load(load_f)
            stance_label.update(load_t['subtaskaenglish'])
            rumor_label.update(load_t['subtaskbenglish'])
        load_f.close()

        # 验证集标签
        with open('./data/RumourEval2019/rumoureval-2019-training-data/dev-key.json', 'r', encoding='utf-8') as load_f:
            load_t = json.load(load_f)
            stance_label.update(load_t['subtaskaenglish'])
            rumor_label.update(load_t['subtaskbenglish'])
        load_f.close()

        for key in rumor_content:
            if key in stance_label:
                rumor_content[key]['label'] = stance_label[key]

            if key in rumor_label:
                rumor_content[key]['r_label'] = rumor_label[key]
  


        #SaveData(CleanText(rumor_content), extra_graph, "semeval2017")
        SaveData(rumor_content, extra_graph, "semeval2019")