import pandas as pd
import numpy as np

data = np.random.random((10, 5))
df = pd.DataFrame(data, index = list(range(1,11)), columns = list(range(1,6)))
df["mean"] = df[df>0.3].mean(axis=1)
print(df)
# print(df.mean(axis=1))
