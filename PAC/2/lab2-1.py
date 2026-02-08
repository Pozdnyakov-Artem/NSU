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

def mult(file,file2):

    f = open(file, "r")

    mat,flag1 = read_mat(f)
    mat2,flag2 = read_mat(f)

    if not( flag1 and  flag2 ):
        print("посторонние символы в матрице или кол-во символов в строках отличается")
        return


    if not(len(mat) != 0 and len(mat2) != 0 and len(mat) == len(mat2[0])):
        print('Нельзя')
        return

    rows = len(mat)
    cols = len(mat2[0])
    inner = len(mat2)
    ans = []

    for k in range(rows):
        row = []

        for i in range(cols):
            el = 0

            for j in range(inner):
                el += (mat[k][j] * mat2[j][i])

            row.append(el)

        ans.append(row)

    f2 = open(file2, "w")

    for i in ans:
        print(' '.join(map(str, i)), file=f2)

    f.close()
    f2.close()

    return




if __name__ == '__main__':

    fopen = input().replace('\\','/')
    fwr = input().replace('\\','/')

    if os.path.isfile(fopen) and os.path.isfile(fwr):
        mult(fopen,fwr)
    else:
        print("Нет таких файлов")