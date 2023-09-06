import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import matplotlib.pyplot as plt
import cv2 as cv


def PSNR(path1, path2):
    img1 = np.array(cv.imread(path1) / 255., dtype=np.float32)
    img2 = np.array(cv.imread(path2)[..., :3]  / 255., dtype=np.float32)
    return compute_psnr(img1, img2)

def PSNR_ip(img1, path2):
    img2 = np.array(cv.imread(path2)[..., :3]  / 255., dtype=np.float32)
    return compute_psnr(img1, img2)

def Show_Diff(path1, path2):
    img1 = np.array(cv.imread(path1) / 255., dtype=np.float32)
    img2 = np.array(cv.imread(path2)/ 255., dtype=np.float32)
    
    diff = np.abs(img1 - img2)
    plt.imsave(f"Diff.png",diff)
    
def resize(path):
    image = np.array(cv.imread(path) / 255., dtype=np.float32)
    return cv.resize(image, (800, 800),interpolation = cv.INTER_AREA)
    



print(f'PSNR = {round(PSNR_ip(resize("./Output.png"), "./Example/Lego_Ref.png"), 4)}')
print(f'PSNR Of Instant-NGP = {round(PSNR("./Example/Lego_NGP.png", "./Example/Lego_Ref.png"), 4)}')