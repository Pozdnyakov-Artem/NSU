from Food import Food

class Inventory:
    def __init__(self,length):
        self._items = [None]*length
        self.length = length

    def add_item(self, item:Food,ind:int):
        if isinstance(item, Food) and isinstance(ind, int) and 0<=ind<self.length:
            if self._items[ind] is None:
                self._items[ind]=item
            elif type(self._items[ind])==type(item):
                self._items[ind].count+=item.count
            else:
                raise TypeError('type conflict')

    def show_items(self):
        print('-' * 10)
        for i in self._items:
            if  not(i is None):
                print(i)
        print('-'*10)

    def take_item(self,ind,count=1):
        if not (self._items[ind] is None) and isinstance(ind,int)and count>0 and 0<=ind<self.length:
            self._items[ind](count)
            if self._items[ind].count==0:
                self._items[ind]=None
        else:
            raise ValueError('Некоректные данные')