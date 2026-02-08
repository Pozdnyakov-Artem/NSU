from abc import abstractproperty, abstractmethod, ABC
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

class Accountant(ABC):

    @staticmethod
    def give_salary(salary,worker):
        if isinstance(worker, Worker):
            worker.take_salary(salary)
        else:
            raise TypeError('worker must be of type Worker')

    @staticmethod
    def print_salary(worker1,worker2):
        if isinstance(worker1, Worker) and isinstance(worker2, Worker):
            print(f'{worker1.name}: {worker1.money}')
            print(f'{worker2.name}: {worker2.money}')
        else:
            raise TypeError('worker must be of type Worker')

class Worker(ABC):
    def __init__(self,name):
        self.name=name
        self.money = 0

    @abstractmethod
    def do_work(self,file):
        pass

    def take_salary(self,salary):
        self.money += salary

    def work(self,file,operation):
        if os.path.exists(file):
            with open(file,'r') as f:
                mat1, flag1 = read_mat(f)
                mat2, flag2 = read_mat(f)

                if flag1 and flag2:
                    if len(mat1)==len(mat2) and len(mat1[0]) == len(mat2[0]):
                        ans=[]
                        for i in range(len(mat1)):
                            new_row=[]
                            for j in range(len(mat1[0])):
                                new_row.append(operation(mat1[i][j],mat2[i][j]))
                            ans.append(new_row)
                        for i in ans:
                            print(*i)
                        print()
                    else:
                        raise ValueError('the matrices do not correspond to each other')
                else:
                    raise TypeError('invalid matrix')

class Pupa(Worker):
    def __init__(self):
        super().__init__('Pupa')

    def do_work(self,file):
        self.work(file,lambda x,y:x+y)


class Lupa(Worker):
    def __init__(self):
        super().__init__('Lupa')

    def do_work(self, file):
        self.work(file,lambda x,y:x-y)


pupa = Pupa()
lupa = Lupa()

Accountant.give_salary(100,pupa)
Accountant.give_salary(10,lupa)

pupa.do_work("matrix.txt")
lupa.do_work("matrix.txt")

Accountant.print_salary(pupa,lupa)

