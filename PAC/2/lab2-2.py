import os

def read_mat(file):

    mat = []
    flag=True

    for i in file:
        new=[]

        if i == "\n":
            break

        for j in i.split():

            if j.isdigit() or (j[0] == '-' and j[1:].isdigit()):
                new.append(int(j))
            else:
                flag=False

        mat.append(new)
    if mat:
        flag=flag and all(len(mat[0]) == len(i) for i in mat)

    return mat,flag

def swert(fopen,fwr):

    f = open(fopen, "r")

    mat1,flag1 = read_mat(f)
    mat2,flag2 =  read_mat(f)

    if not(flag1 and flag2):
        print("посторонние символы в матрице или кол-во символов в строках отличается")
        return

    ans = []
    col1, row1 = len(mat1[0]), len(mat1)
    col2, row2 = len(mat2[0]), len(mat2)

    for i in range(row1 - row2 + 1):
        new = []

        for j in range(col1 - col2 + 1):
            su = 0

            for gor in range(row2):

                for ver in range(col2):
                    su += mat1[i + gor][j + ver] * mat2[gor][ver]

            new.append(su)

        ans.append(new)

    print(*ans, sep='\n')



if __name__ == "__main__":
    fopen=input().replace('\\','/')
    fwr=input().replace('\\','/')
    # fopen = "C://2 sem//PAC//matrix2.txt"
    # fwr = "C://2 sem//PAC//output.txt"

    if os.path.isfile(fopen) and os.path.isfile(fwr):
        swert(fopen,fwr)
    else:
        print("Неверные пути")

