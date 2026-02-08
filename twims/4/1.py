import random
import math
import numpy as np
import matplotlib.pyplot as plt

def drop(la, n):
    p = min(la/n, 1)
    return [random.random() < p for _ in range(n)]

def create_subplots(otr_cent, h, delt, otr_cent2, h2, delt2):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,6))

    ax1.bar(otr_cent, h, width=delt, alpha=0.7,
            color='lightcoral', edgecolor='black')
    ax1.set_title('Ручная гистограмма')
    ax1.set_xlabel('Значения')
    ax1.set_ylabel('Плотность вероятности')

    ax2.bar(otr_cent2, h2, width=delt2, alpha=0.7,
            color='lightcoral', edgecolor='black')
    ax2.set_title('Ручная гистограмма')
    ax2.set_xlabel('Значения')
    ax2.set_ylabel('Плотность вероятности')

    plt.show()

def create_hist(arr, n):
    m = 1 + int(math.log(n, 2))
    # print(m)
    h = []
    otr_cent = []
    delt = (max(arr) - min(arr)) / m
    # print(delt)
    for i in range(m):
        left = min(arr) + i*delt
        right = min(arr) + (i+1)*delt
        count = 0
        for num in arr:
            if left<=num<=right:
                count+=1
        h.append(count/n/delt)
        otr_cent.append((left+right)/2)

    return otr_cent, h, delt

def zad1(la, n, N):

    numb_true = []

    dists = []

    for i in range(N):
        arr = drop(la, n)
        numb_true.append(sum(arr))

        success_positions = [i / n for i, success in enumerate(arr) if success]

        for j in range(1, len(success_positions)):
            dists.append(abs(success_positions[j] - success_positions[j-1]))

    return create_hist(numb_true, len(numb_true)), create_hist(dists, len(dists))


def zad2(la, N):
    dists = []
    for j in range(N):
        X = np.random.poisson(la)
        # print(X)
        points = sorted(np.random.uniform(0, 1, X))
        # print(points)
        for i in range(1,len(points)):
            dists.append(points[i] - points[i-1])
    return create_hist(dists, len(dists))


def zad3(la, N):

    counts = []

    for _ in range(N):
        events = []
        current_time = 0.0

        while current_time <= 1:
            interarrival_time = np.random.exponential(1 / la)
            current_time += interarrival_time

            if current_time <= 1:
                events.append(current_time)
        counts.append(len(events))

    return create_hist(counts, len(counts))


la = 10
n = 10000
N = 10000

(otr_cent, h, delt), (otr_cent2, h2, delt2) = zad1(la, n, N)
otr_cent3, h3, delt3 = zad2(la, N)
otr_cent4, h4, delt4 = zad3(la,N)
create_subplots(otr_cent2, h2, delt2, otr_cent3, h3, delt3)
create_subplots(otr_cent, h, delt, otr_cent4, h4, delt4)
# create_hist([0,7,1,0,-1,6,-1,2,3,4], 10)

