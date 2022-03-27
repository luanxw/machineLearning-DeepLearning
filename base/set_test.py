import numpy as np
# A = np.array(['yGJqTWL6sXmL7eKTUexE', 'yKY7MVvbJQE2Jv2zgTcK' ,'yRHsuwewWFcsudX4AzWx',
#  'yWurRX5SBY54NW5zzDMB' ,'yXHPNL48yagxLNdtxtWm' ,'yanWr3i6SDzqCsMnjbKq'
#  'yjGqPAKATwJdWvghLcJs', 'yu59NsSqitYLdgmYwBWx', 'yvHS5LkKPm5Py5PYeb7A',
#  'zCeeuPGrq9V9YbQotCuq' ,'zN9Sjo35mNpEEjrVaMhG' ])
# B = np.array(['yu59NsSqitYLdgmYwBWx' , 'zTw3cpDpGyX2m5CgECVr',
#  'yGJqTWL6sXmL7eKTUexE', 'yvHS5LkKPm5Py5PYeb7A'])
A=['11@N3', '23@N0', '62@N0','luan']
B=['23@N0', '12@N1','luan']

# # C = [item for sublist in A for item in sublist]
# # D = [item for sublist in B for item in sublist]

# print(A)
# # print(A.size)
# print(B.size)
A= np.array(A)
B= np.array(B)

print(set(B) - (set(A)))
