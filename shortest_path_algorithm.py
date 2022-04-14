import sys
import numba
import numpy
from numba import cuda
import numpy as np
import cs as conn_str
import distance_matrix
import distance_matrix as d
import math

constant_time=2
total_node =1000 #$$$$$$$$$$$$$$$$$
offset=1000 #$$$$$$$$$$$$$$
# d = [[999999, 8, 999999, 5, 999999],
#      [8, 999999, 4, 999999, 10],
#      [999999, 4, 999999, 6, 4],
#      [5, 999999, 6, 999999, 15],
#      [999999, 10, 4, 15,999999]]

# cs = {
#     (1, 2): [0,0,0,1,1,1,0,1,1,1],
#     (2, 3): [0,1,1,1,0,1,1,0,0,0],
#     (1, 4): [1,1,0,1,1,1,0,1,1,0],
#     (4, 3): [0,0,0,0,1,1,0,1,1,0],
#     (2, 5): [0,0,0,1,1,1,1,1,0,0],
#     (3, 5): [1,1,1,0,0,0,0,1,1,0],
#     (4, 5): [0,0,0,0,0,0,0,1,1,1]
# }

cs_adjacency_matrix=[]
distance_adjacency_matrix=[]
adjacency_matrix = {}
path_matrix = {}
final_path_matrix = {}

def create_cs_adjancency_matrix():
    for x in range(total_node):
        for y in range(total_node):
            connectivity_string = total_node*[0]
            if (x + 1, y + 1) in cs:
                connectivity_string = cs[(x + 1, y + 1)]
            elif (y + 1, x + 1) in cs:
                connectivity_string = cs[(y + 1, x + 1)]
            cs_adjacency_matrix.append(connectivity_string)
            distance_adjacency_matrix.append(d[x][y])

# def create_adjacency_matrix(interval1, interval2):
#     temp_adjacency = []
#     for x in range(5):
#         r = []
#         for y in range(5):
#             connectivity_string = [0,0,0,0,0,0,0,0,0,0]
#             if (x + 1, y + 1) in cs:
#                 connectivity_string = cs[(x + 1, y + 1)]
#             elif (y + 1, x + 1) in cs:
#                 connectivity_string = cs[(y + 1, x + 1)]
#             if interval2 > interval1:
#                 t = 2 #interval2 - interval1 + 1
#             else:
#                 t = 2# interval1 - interval2 + 1
#             t = (np.array(connectivity_string,dtype=np.int32), d[x][y], t)
#             r.append(t)
#         temp_adjacency.append(r)
#     adjacency_matrix[(interval1, interval2)] = temp_adjacency

#
# print("Python version:", sys.version)
# print("Numba version:", numba.__version__)
# print("Numpy version:", numpy.__version__)

def convert_to_array(dict):
    return {k:np.array(v) for k,v in dict.items()}

def test():
    return (19,[2,4,5])

def check_if_exist_in_cs(j, q, r):
    try:
        res = ((q, r) in cs and cs[(q, r)][j] == 1) or (
                (r, q) in cs and cs[(r, q)][j] == 1)
    except IndexError:
        res = False
    return res

def find_next_run(p, r, start, finish):
    count = 0
    flag = 0
    j = start
    while j <= 10:
        if check_if_exist_in_cs(j, p, r):
            temp_q = p
            temp_r = r
            if (r, p) in cs:
                temp_q = r
                temp_r = p
            count = count + 1
            #if check_if_adjacency_matrix_exist(adjacency_matrix, temp_q,temp_r) and count == adjacency_matrix[(temp_q, temp_r)][temp_q][temp_r][2]:
            if count == constant_time:
                start = j - constant_time + 1
            if count >= constant_time:
                flag = 1
                finish = j - 1
        else:
            if count >= 2:
                flag = 1
                finish = j - 1
                break
            count = 0

        j = j + 1
    return (start, finish, flag)

def check_if_value_is_greater(param, k, param1, j):
    try:
        return param[k][0] > param1[j][1]
    except IndexError:
        return False

def check_ifpr_i_isequalto_ifqr_k(path_matrix_pr, path_matrix_qr,i,k):
    try:
        return path_matrix_pr[i][1]==path_matrix_qr[k][1]
    except IndexError:
        return False

def path_matrix_exists(path_matrix, p, r, m):
    try:
        return path_matrix[p*offset+r][m]
        True
    except IndexError:
        return False

def is_vpr_greater_than_zero(param):
    try:
        return param[0]>0
    except IndexError:
        return False

@cuda.jit
def find_minimum_time_path(path_matrix):
    blockDim = cuda.blockDim.x
    blockId = cuda.blockIdx.x
    tx = (blockId * blockDim + cuda.threadIdx.x) + 1

    ty = (blockId * blockDim + cuda.threadIdx.y) + 1
    p=tx
    r=ty
    if p<total_node+1 and r<total_node+1 and p!=r:
        if is_vpr_greater_than_zero(path_matrix[p*offset+r]):
            ifpr = path_matrix[p*offset+r][1][1]
            ispr = path_matrix[p*offset+r][1][0]
            pred = path_matrix[p*offset+r][1][2]
            min_time = ifpr - ispr + 1
            temp_start = ispr
            temp_finish = ifpr
            temp_predecessor = pred
            previous_node = pred
            for j in range(2, path_matrix[p*offset+r][0], 1):
                temp_duration = path_matrix[p*offset+r][j][1] - path_matrix[p*offset+r][j][0] + 1
                if temp_duration < min_time:
                    min_time = temp_duration
                    temp_start = path_matrix[p*offset+r][j][0]
                    temp_finish = path_matrix[p*offset+r][j][1]
                    temp_predecessor = path_matrix[p*offset+r][j][2]

            vpr = 1
            ispr = temp_start
            ifpr = temp_finish
            pred = temp_predecessor
            temp_path_matrix_list = list(path_matrix[p*offset+r][1])
            temp_path_matrix_list[0] = ispr
            temp_path_matrix_list[1] = ifpr
            temp_path_matrix_list[2] = pred
            path_matrix[p*offset+r][1] = tuple(temp_path_matrix_list)
            path_matrix[p*offset+r][0] = vpr

@cuda.jit
def update_all_possible_contacts_cuda(a):
    #print(a)
    blockDim = cuda.blockDim.x
    blockId = cuda.blockIdx.x
    tx = (blockId * blockDim + cuda.threadIdx.x) + 1

    ty = (blockId * blockDim + cuda.threadIdx.y) + 1
    p=tx
    r=ty
    if p<=total_node and r<=total_node and p!=r:
        for q in range(total_node):
            q = q + 1
            if q != r:
                if p*offset+q in path_matrix and q*offset+r in path_matrix and p*offset+r in path_matrix:
                    vpq = path_matrix[p*offset+q][0]
                    vqr = path_matrix[q*offset+r][0]
                    vpr = path_matrix[p*offset+r][0]
                    if vpq > 0 and vqr > 0:
                        k = 1
                        m = path_matrix[p*offset+r][0]
                        while k <= vqr:
                            flag1 = 0
                            j = vpq
                            while j >= 1 and flag1 == 0:
                                if check_if_value_is_greater(path_matrix[q*offset+r], k, path_matrix[p*offset+q], j):
                                    flag2 = 0
                                    for i in range(10):
                                        i = i + 1
                                        if check_ifpr_i_isequalto_ifqr_k(path_matrix[p*offset+r], path_matrix[q*offset+r], i, k):
                                            if path_matrix[p*offset+r][i][0] < path_matrix[p*offset+q][j][0]:
                                                temp_path_matrix_list = list(path_matrix[p*offset+r][i])
                                                temp_path_matrix_list[0] = path_matrix[p*offset+q][j][0]
                                                # path_matrix[(p, r)][i][0]=path_matrix[(p,q)][j][0]
                                                # path_matrix[(p, r)][i][2] = q
                                                temp_path_matrix_list[2] = q
                                                path_matrix[p*offset+r][i] = tuple(temp_path_matrix_list)
                                            flag2 = 1
                                        else:
                                            flag2 = 0
                                    if flag2 == 0:
                                        m = m + 1
                                        if path_matrix_exists(path_matrix, p, r, m):
                                            temp_path_matrix_list = list(path_matrix[p*offset+r][m])
                                            temp_path_matrix_list[0] = path_matrix[p*offset+q][j][0]
                                            temp_path_matrix_list[1] = path_matrix[q*offset+r][k][1]
                                            temp_path_matrix_list[2] = q
                                            path_matrix[p*offset+r][m] = tuple(temp_path_matrix_list)
                                            flag1 = 1
                                            path_matrix[p*offset+r][0] = m
                                j = j - 1
                            k = k + 1
                        vpr = m
@cuda.jit
def initialize_path_matrix_cuda(a,b):
    blockDim = cuda.blockDim.x
    blockId = cuda.blockIdx.x
    tx = (blockId*blockDim+cuda.threadIdx.x)+1

    ty=(blockId*blockDim+cuda.threadIdx.y)+1
    #if tx == 9:
        #print("yes")
    if tx==9 and ty == 33:
        print('yes')
    if(tx<=total_node and ty<=total_node):
        blockDim = cuda.blockDim.x
        blockId = cuda.blockIdx.x
        #tx = blockId*blockDim + tx

        if(tx<=total_node and ty<=total_node):
            ispr={1:0}
            ifpr={1:0}
            if(tx==ty):
                predpr ={1:0}
                path_matrix[tx*offset+ty]=[0]
                path_matrix[tx * offset + ty].append((ispr[1],ifpr[1],predpr[1]))

            else:
                start = 0
                finish = 0
                k = 0
                #print("here inside else")
                while k < 10 - 2:  # -adjacency_matrix[(p+1,r+1)][p+1][r+1][2]:  # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    # removed P
                    res = find_next_run(tx, ty, start, finish)
                    start = res[0]
                    finish = res[1]
                    flag = res[2]
                    if flag == 1:
                        templist = []
                        temp = start + constant_time - 1
                        while temp <= finish:
                            k = k + 1
                            ispr[k] = start
                            ifpr[k] = temp
                            start = start + 1
                            temp = temp + 1
                            #print(str(tx) + " " + str(ty) + ":" + str(ispr[k]) + " " + str(ifpr[k]) + " " + str(tx))

                        vpr = k
                        templist.insert(0, vpr)
                        path_matrix[tx * offset + ty] = templist

                        for i in range(vpr):
                            path_matrix[tx * offset + ty].append((ispr[i + 1], ifpr[i + 1], tx))
                        #print(path_matrix)
                    else:
                        # added
                        k = k + 1
                        if tx * offset + ty not in path_matrix:
                            path_matrix[tx * offset + ty] = [0, (0, 0, 999999)]
    #print(path_matrix[11])
    #print(path_matrix)
    S=a
    d=[[1,2,3],[4,5,6]]
    d.append([[7,8,9],[1,2,3]])
    q={1:0}
    #print(test())
    #for i in range(array.size):
        #array[i] += 0.5
def final_path_matrix_cal():
    
    with open("F:\\project1\\output" + ".txt", 'w') as file:  # file.write(str(cs))
        for i in range(total_node):
            i=i+1
            for j in range(total_node):
                j=j+1
                if i*offset+j in path_matrix:
                    final_path_matrix[i*offset+j]=path_matrix[i*offset+j][1]
                    #print(str(final_path_matrix[i*offset+j]) + "  ", end=" ")
                    #file.write(str(final_path_matrix[i*offset+j]))
            #print("\n")
            #file.write("\n")
        file.write(str(final_path_matrix))
if __name__ == '__main__':
    cs = conn_str.conn_string
    d=distance_matrix.d_m
    create_cs_adjancency_matrix()
    # for x in range(5):
    #     for y in range(5):
    #         create_adjacency_matrix(x + 1, y + 1)
    # a=np.array(list(adjacency_matrix.values()))
    # #a=a.astype(np.float)
    # #a_g = cuda.to_device(a)
    # aa = numpy.array([[2,0,4,4],[1,3,3,2],[5,7,2,6]])
    initialize_path_matrix_cuda[10, (32,32)](cs_adjacency_matrix, distance_adjacency_matrix)
    # cudakernel0[6, (128,128)](cs_adjacency_matrix,distance_adjacency_matrix)
    print(path_matrix)
    update_all_possible_contacts_cuda[10,(32,32)](path_matrix)
    print(path_matrix)
    find_minimum_time_path[10,(32,32)](path_matrix)
    print(path_matrix)
    final_path_matrix_cal()
    #cudakernel0[6, (128,128)](cs_adjacency_matrix,distance_adjacency_matrix)
    #print(convert_to_array(adjacency_matrix))

# array = numpy.ones(2)
# print('Initial array:', array)
#
# print('Kernel launch: cudakernel0[1, 1](array)')
# cudakernel0[6, 128](array)
#
# print('Updated array:',array)
