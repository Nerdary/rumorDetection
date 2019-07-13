import numpy as np
import matplotlib.pyplot as plt

c_c = [76.37, 30.97, 66.31]
d_c = [6.97, 8.02, 6.94]
d_c_1 = [83.34, 38.99, 73.25]
c_d = [5.68, 7.83, 11.4]
c_d_1 = [89.02, 46.84, 84.65]
q_c = [0, 9.24, 3.31]
q_c_1 = [89.02, 56.08, 87.96]
c_q = [0, 8.46, 0]
c_q_1 = [89.02, 64.54, 87.96]
c_s = [0, 7.7, 0]
c_s_1 = [89.02, 72.24, 87.96]
s_c = [0, 7.67, 4.58]
s_c_1 = [89.02, 79.91, 92.54]
d_q = [0, 6.64, 0]
d_q_1 = [89.02, 86.55, 92.54]
s_d = [0, 5.19, 0]
s_d_1 = [89.02, 91.74, 92.54]
q_s = [0, 5.07, 0]

ind = np.arange(3)
plt.figure(figsize=(5, 2.5))
p1 = plt.bar(ind, c_c, 0.5, color='r')
p2 = plt.bar(ind, d_c, 0.5, bottom=d_c_1, color='b', hatch='*')
p3 = plt.bar(ind, c_d, 0.5, bottom=c_d_1, color='coral', hatch='o')

plt.ylabel('Frequency')
plt.xticks(ind, ('c-c', 'd-c', 'c-d'))
plt.legend((p1[0], p2[0], p3[0]), ('true', 'unverified', 'deny'), fontsize=12)

plt.show()