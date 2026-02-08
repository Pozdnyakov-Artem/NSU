import numpy as np

def mov_vec(vector,wind_size):

    vec=np.array(vector,dtype=float)

    return np.convolve(vec,np.ones(wind_size)/wind_size,mode='valid')

vector = [1,2,3,4,5,6,7,8,9,10]

print(mov_vec(vector,4))