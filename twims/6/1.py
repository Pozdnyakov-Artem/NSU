import numpy as np
import pandas as pd
import scipy.stats as stats

def cor(cl1, cl2):
    return (cl1.size * (cl1*cl2).sum() - cl1.sum()*cl2.sum()) / np.sqrt((cl1.size * (cl1**2).sum() - cl1.sum()**2) * (cl2.size * (cl2**2).sum() - cl2.sum()**2))

def lin(df, arr):
    y = df[arr[0]].values.reshape(-1, 1)
    const_col = np.ones((len(y), 1))

    if len(arr) > 1:
        other_cols = df[arr[1:]].values
        x = np.hstack([const_col, other_cols])
    else:
        x = const_col

    mask = ~np.any(np.isnan(x), axis=1) & ~np.isnan(y.flatten())
    x=x[mask]
    y=y[mask]

    fst = x.transpose()@x
    b = np.linalg.inv(fst) @ x.transpose()@y #вектор коэф
    yt = x@b                                 #pred знач
    e = y - yt                               #ошибки
    sig = (e.transpose()@e) / (len(y)-x.shape[1]) #дисп
    return np.linalg.inv(fst), b, e.flatten(), sig

def zad3(xtx, sig, ind, b_old, df):
    b = sig * xtx[ind, ind] #дисп
    se = np.sqrt(b)         #st ошиб

    t_s = b_old[ind] / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_s), df))

    return b, se, p_value



df = pd.read_csv(R'C:\2 sem\twims\6\House-Prices - House-Prices.csv')

dummies = pd.DataFrame()
for i in df.columns:
    if i not in ["price","airport", "waterbody", "bus_ter"]:
        if abs(cor(df["price"], df[i])) > 0.9:
            df.drop([i], axis=1, inplace = True)
    elif i != "price":
        dummies = pd.concat([dummies, pd.get_dummies(df[i], prefix=f"{i}", dtype=int, drop_first=True),], axis=1)
        df.drop(i, axis=1, inplace=True)

func = [np.sqrt, lambda x: np.log(x + 1 - x.min()) if x.min() <= 0 else np.log(x), np.exp, lambda x: x**2]

for i in df.drop(["price"],axis=1).columns:
    cors =np.array([cor(df["price"], df[i])])
    for j in func:
        cors = np.append(cors, abs(cor(df["price"], j(df[i]))))
    df[i] = df[i] if np.argmax(cors) == 0 else func[np.argmax(cors)-1](df[i])

df = pd.concat([df, dummies], axis = 1)

model = lin(df, df.columns)
print(model)
print(df.size)
# print(df)
for i, col in enumerate(df.columns):
    print("незначимое") if zad3(model[0], model[3], i, model[1], len(df)-len(model[1]))[2] > 0.5 else print("значимое")
