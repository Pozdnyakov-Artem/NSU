import os
import cv2
import numpy as np

def aug(img, type_oper):
    match type_oper:
        case 0:
                rat_mat = cv2.getRotationMatrix2D([250,250],np.random.randint(1,359),1)
                return cv2.warpAffine(img, rat_mat, (500,500))
        case 1:
                return cv2.flip(img, np.random.choice([-1,0,1]))
        case 2:
                x1 = np.random.randint(0,400)
                x2 = np.random.randint(x1+1, 500)
                y1 = np.random.randint(0,400)
                y2 = np.random.randint(y1+1, 500)

                return img[x1:x2, y1:y2]
        case 3:
                return cv2.medianBlur(img, 3)
        case _:
                return img

def gen_img_and_lbl(kol):

    def gen(kol):
        images = sorted(os.listdir(r"C:\2 sem\PAC\5\images"))
        labels = sorted(os.listdir(r"C:\2 sem\PAC\5\labels"))

        mas = list(zip(images, labels))

        np.random.shuffle(mas)

        start = 0
        end = kol

        while True:
            yield mas[start:end]
            if end+kol < len(mas):
                start = end
                end = end + kol
            else:
                start = 0
                end = kol

    shag = gen(kol)
    while True:
        iter = next(shag)
        img=[]
        lbl=[]
        for i in range(kol):
            ind = np.random.randint(0, 4)
            img.append(aug(cv2.resize(cv2.imread(r"C:\2 sem\PAC\5\images\\"+iter[i][0],1),(500,500)), ind))
            lbl.append(aug(cv2.resize(cv2.imread(r"C:\2 sem\PAC\5\labels\\"+iter[i][1],1),(500,500)), ind))

        yield img, lbl

try:
    gen_img, gen_lbl = next(gen_img_and_lbl(1))
except:
    print("Конец")