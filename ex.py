data = [[[0, 1, 2, 3, 4], [11, 22, 33, 44, 55]]]#, ['qq', 'ww', 'ee', 'rr', 'tt']]]
# for idx, data in enumerate(data):
#     print('1loop', data)
#     for idx, adm in enumerate(data):
#         seq_input = data[:idx+1]
#         print('2loop', seq_input, idx)
#         for adm2 in seq_input:
#             print(adm2)
            
# for step, input in enumerate(data):
#     # input -> [[환자1의 ICD9_CODE 인덱스], [환자1의 PRO_CODE 인덱스], [환자1의 NDC 인덱스]]
#     for idx, adm in enumerate(input):
#         print(idx)
#         # 첫번째는 ICD9_CODE 만, 두번째는 PRO_CODE 포함, 세번째는 전체.
#         seq_input = input[:idx+1]     
#         for adm2 in seq_input:
#             print(adm2[0], adm2[1])

import torch
import torch.nn as nn
import numpy as np
# embedding = nn.Embedding(1000, 64)
# print(embedding(torch.LongTensor([10])).shape)
# embedding = embedding(torch.LongTensor([10]).unsqueeze(dim=0))
# print(embedding.shape)
# embedding = embedding.mean(dim=1).unsqueeze(dim=0)
# print(embedding.shape)


# zero = np.zeros((0, 10))
# print(zero)
# zero = np.zeros((1, 10))
# print(zero)

# loss1_target = np.zeros((1, 10))
# loss1_target[:, [0,1,2,3,4]] = 1
# loss3_target = np.full((1, 10), -1)
# for idx, item in enumerate([0,1,2,3,4]):
#     loss3_target[0][idx] = item

# print(loss1_target)
# print(loss3_target)

# ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
# print(ja, prauc, avg_p, avg_r, avg_f1)

y_gt = torch.Tensor([1,0,1,0,0,0,1])
y_pred = torch.Tensor([0,0,0,0,0,1,1])
target = np.where(y_gt == 1)[0]
out_list = np.where(y_pred == 1)[0]
inter = set(out_list) & set(target)
union = set(out_list) | set(target)
jaccard_score = 0 if union == 0 else len(inter) / len(union)
print(jaccard_score)