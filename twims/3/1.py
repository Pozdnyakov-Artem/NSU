import numpy as np

def mad(data):
    median = np.median(data)

    new_data = abs(data - median)

    return np.median(new_data)

def disp(data):
    mean = np.mean(data)

    return sum((data - mean)**2)/len(data)

def boot_strap(data,kol_strapov):

    n = len(data)

    mas_mad = np.array([], dtype = np.float32)
    mas_disp = np.array([], dtype = np.float32)

    for _ in range(kol_strapov):

        boot_stra = np.sort(np.random.choice(data,size=n,replace=True))

        mas_mad = np.append(mas_mad, mad(boot_stra))

        mas_disp = np.append(mas_disp, disp(boot_stra))

    mas_mad = np.sort(mas_mad)

    mas_disp = np.sort(mas_disp)

    low = int(0.005 * kol_strapov)
    high = kol_strapov - low

    return low, high, mas_mad, mas_disp

X = np.random.normal ( scale =1.0 , size =10000)

low, high, mas_mad, mas_disp = boot_strap(X,1000)

mad_ci_lower, mad_ci_upper = mas_mad[low], mas_mad[high]

disp_ci_lower, disp_ci_upper = mas_disp[low], mas_disp[high]

mad_point = mad(X)
var_point = disp(X)

print("=== ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ===")
print(f"MAD:")
print(f"  Точечная оценка: {mad_point:.4f}")
print(f"  95% ДИ: [{mad_ci_lower:.4f}, {mad_ci_upper:.4f}]")
print(f"  Ширина интервала: {mad_ci_upper - mad_ci_lower:.4f}")

print(f"\nДисперсия:")
print(f"  Точечная оценка: {var_point:.4f}")
print(f"  95% ДИ: [{disp_ci_lower:.4f}, {disp_ci_upper:.4f}]")
print(f"  Ширина интервала: {disp_ci_upper - disp_ci_lower:.4f}")