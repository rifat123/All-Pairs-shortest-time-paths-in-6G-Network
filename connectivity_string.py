snapshot1 = [
    [0, 1, 0, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1],
    [1, 1, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
snapshot2 = [
    [0, 1, 0, 0, 1],
    [1, 1, 0, 1, 1],
    [0, 0, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
# for i in snapshot1,snapshot2:
# print(i)
#file=open("cs_full_data_set.txt","w")

import json
with open("0.txt") as f:
    ss0 = json.load(f)
with open("1.txt") as f:
    ss1 = json.load(f)

with open("2.txt") as f:
    ss2 = json.load(f)

with open("3.txt") as f:
    ss3 = json.load(f)

with open("4.txt") as f:
    ss4 = json.load(f)
with open("5.txt") as f:
    ss5 = json.load(f)
with open("6.txt") as f:
    ss6 = json.load(f)
with open("7.txt") as f:
    ss7 = json.load(f)

with open("8.txt") as f:
    ss8 = json.load(f)
with open("9.txt") as f:
    ss9 = json.load(f)

cs = {}
#with open("F:\\project1\\output"+".txt", 'w') as file:
for i in range(1000):
    for j in range(1000):

        # offset=i*5+j
        if (i != j) :

            for k in ss0,ss1,ss2,ss3,ss4,ss5,ss6,ss7,ss8,ss9:
                #print(k)
                if k[i][j] == 1:
                    if (i+1,j+1) in cs:
                        cs[(i+1,j+1)].append(1)
                        #file.write(str(cs[(i+1,j+1)]))
                    else:
                        cs[(i+1, j+1)] = [1]
                        #file.write(str(cs[(i + 1, j + 1)]))

                    # print('Here')
                else:
                    if (i+1, j+1) in cs:
                        cs[(i+1, j+1)].append(0)
                        #file.write(str(cs[(i + 1, j + 1)]))
                    else:
                        cs[(i+1, j+1)] = [0]
                        #file.write(str(cs[(i + 1, j + 1)]))
                    # print('Here 2')
        else:
            cs[(i+1, j+1)] = 10*[0]
            #file.write(str(cs[(i + 1, j + 1)]))

with open("F:\\project1\\output" + ".txt", 'w') as file:       #file.write(str(cs))
    file.write(str(cs))
    file.close()