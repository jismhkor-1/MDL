# from xml.etree import ElementTree as ET
#
#
# tree = ET.parse('../archive/Restaurants_Train_v2.xml')
# root = tree.getroot()
#
# record = []
# catList = set()
# sentList = set()
# c = 0
# for node in root:
#     t = node.find("text")
#     text = t.text
#     acs = node.find("aspectCategories")
#     cs = []
#     flag = 0
#     for item in acs.findall("aspectCategory"):
#         att = item.attrib
#         catList.add(att['category'])
#         sentList.add(att['polarity'])
#         if att['polarity'] == 'conflict':
#             flag = 1
#         cs.append(att)
#     if flag == 1:
#         continue
#     if len(cs) > 1:
#         c += 1
#     record.append({"text": text, "category_sentiment": cs})
#
# print(len(catList))
# print(catList)
# print(len(sentList))
# print(sentList)
# print(len(record))
# print(c)


import json


f = open('../MAMS/MAMS_train.txt')
record = []
cList = set()
pList = set()
for line in f.readlines():
    x = line.strip().split('\001')
    text = x[0]
    c = x[1].split()[4]
    s = x[1].split()[6]
    cList.add(c)
    pList.add(s)
    record.append({"text": text, "cs": {"category": c, "polarity": s}})
f.close()
print(len(record))

cList = list(cList)
cList.sort()
c2id = {c: i for i, c in enumerate(cList)}
s2id = {"positive": 0, "neutral": 1, "negative": 2}

dataset = []
tmp = record[0]['text']
cs_ = record[0]['cs']
cs_['category_id'] = c2id[cs_['category']]
cs_['polarity_id'] = s2id[cs_['polarity']]
cs = [cs_]
for r in record[1:]:
    t = r['text']
    if t == tmp:
        cs_ = r['cs']
        cs_['category_id'] = c2id[cs_['category']]
        cs_['polarity_id'] = s2id[cs_['polarity']]
        cs.append(cs_)
    else:
        dataset.append({"text": tmp, "category_sentiment": cs})
        tmp = t
        cs_ = r['cs']
        cs_['category_id'] = c2id[cs_['category']]
        cs_['polarity_id'] = s2id[cs_['polarity']]
        cs = [cs_]
dataset.append({"text": tmp, "category_sentiment": cs})
json.dump(dataset, open('../MAMS/train.json', 'w', encoding='utf-8'), indent=2)
