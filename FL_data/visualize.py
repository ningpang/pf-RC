import json
import numpy as np
import matplotlib.pyplot as plt
# data = json.load(open('./tacred/niid_label/8-10/train.json', 'r'))
# rel2temp = json.load(open('./tacred/rel2id.json', 'r'))
import torch
import torch.nn.functional as F
import seaborn as sns

data = json.load(open('./tacred/niid_quantity/8-100/train.json', 'r'))
rel2temp = json.load(open('tacred/rel2id.json', 'r'))
rel2id = {}

for rel in rel2temp:
    rel2id[rel] = len(rel2id)
counter = torch.zeros((len(data), len(rel2id)))

for client_id in range(len(data)):
    for rel in data[client_id]:
        counter[client_id][rel2id[rel]] = len(data[client_id][rel])

counter = F.softmax(counter, dim=0)

# x = list(range(len(data)))
# y = list(range(len(rel2id)))
#
# for i in range(len(x)):
#     for j in range(len(y)):
#         print(i, j)
#         plt.scatter(x[i], y[j], color='b', marker='o', s=counter[i][j])

# plt.imshow(counter, cmap=plt.cm.blue, vmin=0, vmax=1)
# plt.colorbar()
# plt.show()
plt.figure()
sns.heatmap(counter, fmt='d')
plt.show()