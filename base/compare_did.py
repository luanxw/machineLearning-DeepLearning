import numpy as np
import json 


# paas_did = np.loadtxt('../machineLearning/data/paas_did.txt' ,dtype=str,delimiter=',')
# saas_did = np.loadtxt('../machineLearning/data/saas_did.txt' ,dtype=str,delimiter=',')
# print('paas 层所有的did:' ,paas_did.shape)
# print('saas 层所有的did:' ,saas_did.shape)

# differrent  = list(set(paas_did) - set(saas_did)) 
# # print('pasa为主的差集', differrent)
# print('pasa为主的差集', differrent)

# with open('../machineLearning/data/different_did.txt','w') as wf:
#     for i in differrent:
#      wf.writelines(i)
#      wf.writelines('\r\n')
# wf.close()

# print('save success!!!', np.array(differrent))


# saas_data = []
# with open('../machineLearning/data/saas.json','r') as jf:
#     saas_data = json.load(jf)

# saas_did = []
# for item in saas_data:
#     saas_did.append(item.get('did'))
# saas_did = np.array(saas_did)


# print(saas_did.shape)
# with open('../machineLearning/data/saas_did.txt','w') as wf:
#     for i in saas_did:
#      wf.writelines(i)
#      wf.write('\r\n')
# wf.close()


# print(paas_did.shape)
# print(saas_did.shape)
# print(paas_did)
# print(saas_did)
# union_did = list(set(paas_did).intersection(set(saas_did)))
# print(union_did)

# union_did = np.array(union_did)
# print(union_did.shape)
# paas_alone = list(set(paas_did).difference(set(saas_did))) # paas_did中有而saas_did中没有的
# print(np.array(paas_alone).shape)
# all_did = list(set(paas_did) | set(saas_did)) 
# print('并集',np.array(all_did).shape)

# cha = list(set(paas_did) - set(saas_did)) 
# print('pasa为主的差集',np.array(cha).shape)

# cha2 = list(set(saas_did) - set(paas_did)) 
# print('saas为主的差集',np.array(cha2).shape)



#    gHkn68YhJXKDAD4hRQzNWb

# paas_did = np.loadtxt('../machineLearning/data/paas_did_t.txt' ,dtype='S',delimiter=',')
# saas_did = np.loadtxt('../machineLearning/data/saas_did_t.txt' ,dtype='S',delimiter=',')
# print(paas_did)
# print(saas_did)
# paas_alone = list(set(paas_did)&set(saas_did)) # paas_did中有而saas_did中没有的
# print('交集',np.array(paas_alone))

# # print(set(paas_did))
# all_did = list(set(paas_did) | set(saas_did)) 
# print('并集',np.array(all_did))

# cha = list(set(paas_did) - set(saas_did)) 
# print('pasa为主的差集',np.array(cha))

# cha2 = list(set(saas_did) - set(paas_did)) 
# print('saas为主的差集',np.array(cha2))

extra_did = np.loadtxt('../machineLearning/data/Extra_did.txt' ,dtype=str,delimiter=',')
different_did = np.loadtxt('../machineLearning/data/different_did.txt' ,dtype=str,delimiter=',')
print('extra_did 中did数量: ',extra_did.shape)
print('different_did 中did数量: ',different_did.shape)

result = list(set(different_did) - set(extra_did))
print('different_did 独有的did: ',np.array(result).shape)


result2 = list(set(extra_did) - set(different_did))
print('extra_did 独有的did: ',np.array(result2).shape)

with open('../machineLearning/data/delete_did.txt','w') as wf:
    for i in result:
     wf.writelines(i)
     wf.writelines('\r\n')
wf.close()
print('导出成功!!!')