import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

input = torch.load('0.1_noise.pt').to('cpu')
input = input[10,:]
targets = torch.load('0.1_gt.pt').to('cpu')
targets = targets[10,:]

# resolution
re_sp = 0.1         # 0.1m
re_de = 0.04        # 0.04m

# length
l_sp = 2
l_de = 1

# starting position
x_sp = 1
y_sp = -1
x_de = 0.5
y_de = -0.5

sparse_x = []
sparse_y = []
sparse_z = []

dense_x = []
dense_y = []
dense_z = []

#######################################################################################
# sparse 저장
#######################################################################################
input_x = x_sp
input_y = y_sp

# x값 저장
for i in range(int(l_sp/re_sp) + 1):         # y 변화
    for j in range(int(l_sp/re_sp) + 1):     # x 변화
        sparse_x.append(input_x)
        sparse_y.append(input_y)
        sparse_z.append(targets[j + i*(int(l_sp/re_sp) + 1)])
        
        input_y = input_y + re_sp
    input_y = y_sp
    input_x = input_x - re_sp

#######################################################################################
# dense 저장
#######################################################################################
input_x = x_de
input_y = y_de

# x값 저장
for i in range(int(l_de/re_de) + 1):         # y 변화
    for j in range(int(l_de/re_de) + 1):     # x 변화
        dense_x.append(input_x)
        dense_y.append(input_y)
        dense_z.append(targets[j + i*(int(l_de/re_de) + 1)])
        
        input_y = input_y + re_de
    input_y = y_de
    input_x = input_x - re_de


# 그래프 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# sparse 점 추가
ax.scatter(sparse_x, sparse_y, sparse_z, c='green', marker='o', label='sparse')

# dense 점 추가
ax.scatter(dense_x, dense_y, dense_z, c='red', marker='^', label='dense')

# 축 라벨
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("High noise level 3D Scatter Plot")

# 범례 추가
ax.legend()

plt.savefig("0.1_targets_scatter_plot.png")