#!/usr/bin/python
#-*-coding:utf-8-*-
import json
import numpy as np
import matplotlib.pyplot as plt

extra_graph = []
rumor_content = dict()
pair = {'true': dict(), 'unverified': dict(), 'false': dict()}
stat = {'true': {'comment': set(), 'deny': set(), 'query': set(), 'support': set()},
        'unverified': {'comment': set(), 'deny': set(), 'query': set(), 'support': set()},
        'false': {'comment': set(), 'deny': set(), 'query': set(), 'support': set()}}

def Extract_Pair(g, c, label, last):
    res = []
    l = ''
    for k in g:
        if label == 'null':
            if k in rumor_content:
                l = rumor_content[k]['r_label']
        else:
            l = label

        for kk in g[k]:
            if last != 0:
                res.append([k, kk, l])
            if type(g[k][kk]) == dict:
                res += Extract_Pair(g[k], c, l, 1)

    return res

f = open('./data/pdata/semeval2017.extra.g.dict', 'r', encoding='utf-8')
extra_graph = eval(f.read())
f.close()

f = open('./data/pdata/semeval2017.rumor.dict', 'r', encoding='utf-8')
rumor_content = eval(f.read())
f.close()

r = Extract_Pair(extra_graph, rumor_content, 'null', 0)

for x in r:
    if x[0] in rumor_content and x[1] in rumor_content:
        stat[x[2]][rumor_content[x[0]]['label']].add(x[0])
        stat[x[2]][rumor_content[x[1]]['label']].add(x[1])
        item = "{}-{}".format(rumor_content[x[0]]['label'], rumor_content[x[1]]['label'])
        if item in pair[x[2]]:
            pair[x[2]][item] += 1
        else:
            pair[x[2]][item] = 0

statnei = {'comment': {'comment': 0, 'deny': 0, 'query': 0, 'support': 0},
           'deny': {'comment': 0, 'deny': 0, 'query': 0, 'support': 0},
           'query': {'comment': 0, 'deny': 0, 'query': 0, 'support': 0},
           'support': {'comment': 0, 'deny': 0, 'query': 0, 'support': 0},}

show_item = set()
for k in pair:
    print(k)
    sum = 0
    for kk in pair[k]:
        sum += pair[k][kk]

    del_item = []
    for kk in pair[k]:
        pair[k][kk] = round(pair[k][kk]*100/sum, 2)
        # if pair[k][kk] < 3:
        #         del_item.append(kk)

    # for item in del_item:
    #     pair[k].pop(item)

    print(sorted(pair[k].items(), key=lambda d: d[1], reverse=True))
    for kk in pair[k]:
        words = kk.split('-')
        statnei[words[0]][words[1]] += pair[k][kk]

d = []
c = []
d = []
q = []
s = []
for k in statnei:
    print(k)
    print(sorted(statnei[k].items(), key=lambda d: d[1], reverse=True))
    sum = statnei[k]['comment'] + statnei[k]['query'] + statnei[k]['deny'] + statnei[k]['support']
    c.append(round(statnei[k]['comment']*100/sum, 2))
    q.append(round(statnei[k]['query']*100/sum, 2))
    d.append(round(statnei[k]['deny']*100/sum, 2))
    s.append(round(statnei[k]['support']*100/sum, 2))

print(c)
print(q)
print(d)
print(s)
#
# qq = []
# ddd = []
#
# for i in range(len(d)):
#     qq.append(c[i]+q[i])
#     ddd.append(c[i]+q[i]+d[i])
#
# # for k in stat:
# #     x = []
# #     for kk in stat[k]:
# #         x.append(len(stat[k][kk]))
# #
# #     print(round(x[0]/sum(x[0:4]), 4), round(x[1]/sum(x[0:4]), 4), round(x[2]/sum(x[0:4]), 4), round(x[3]/sum(x[0:4]), 4))
# #
# ind = np.arange(4)
# plt.figure(figsize=(5, 2.5))
# p1 = plt.bar(ind, c, 0.5, color='r', hatch='-')
# p2 = plt.bar(ind, q, 0.5, bottom=c, color='g', hatch='*')
# p3 = plt.bar(ind, d, 0.5, bottom=qq, color='b', hatch='o')
# p4 = plt.bar(ind, s, 0.5, bottom=ddd, color='y', hatch='+')
#
# plt.ylabel('Frequency')
# plt.xticks(ind, ('comment', 'deny', 'query', 'support'))
# plt.legend((p1[0], p2[0], p3[0], p4[0]), ('comment', 'query', 'deny', 'support'), fontsize=12)
#
# plt.show()