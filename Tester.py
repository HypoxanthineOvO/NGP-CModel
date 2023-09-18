import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import matplotlib.pyplot as plt
import cv2 as cv

def PSNR(path1, path2):
    img1 = np.array(cv.imread(path1) / 255., dtype=np.float32)
    img2_raw = cv.imread(path2, cv.IMREAD_UNCHANGED) / 255.
    img2_raw = img2_raw[..., :3] * img2_raw[..., 3:]
    img2 = np.array(img2_raw, dtype=np.float32)
    return compute_psnr(img1, img2)

def PSNR_ip(img1, path2):
    img2_raw = cv.imread(path2, cv.IMREAD_UNCHANGED) / 255.
    img2_raw = img2_raw[..., :3] * img2_raw[..., 3:]
    img2 = np.array(img2_raw, dtype=np.float32)
    return compute_psnr(img1, img2)

def Show_Diff(path1, path2, name = None):
    img1 = np.array(cv.imread(path1) / 255., dtype=np.float32)
    img2_raw = cv.imread(path2, cv.IMREAD_UNCHANGED) / 255.
    img2_raw = img2_raw[..., :3] * img2_raw[..., 3:]
    img2 = np.array(img2_raw, dtype=np.float32)
    
    diff = np.abs(img1 - img2)
    out_name = "Diff"
    if name is not None:
        out_name = name
    plt.imsave(f"{out_name}.png",diff)
    
def resize(path):
    image = np.array(cv.imread(path) / 255., dtype=np.float32)
    return cv.resize(image, (800, 800),interpolation = cv.INTER_AREA)
# ./data/nerf_synthetic/lego/test/r_0.png -> Lego_Ref

if __name__ == "__main__":
    name = "chair"
    print(f'PSNR: \t{round(PSNR_ip(resize("./Output.png"), f"./data/nerf_synthetic/{name}/test/r_0.png"), 4)}')
    Show_Diff("./Output.png", f"./data/nerf_synthetic/{name}/test/r_0.png", f"Diff_{name}")
