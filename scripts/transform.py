import numpy as np
import torch
import json, time, os, msgpack
from tqdm import tqdm


def Part_1_By_2(x: np.ndarray):
    x &= 0x000003ff;                 # x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff # x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x <<  8)) & 0x0300f00f # x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x <<  4)) & 0x030c30c3 # x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x <<  2)) & 0x09249249 # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x

def morton_naive(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    return Part_1_By_2(x) + (Part_1_By_2(y) << 1) + (Part_1_By_2(z) << 2)

def morton(input):
    return morton_naive(input[..., 0], input[..., 1], input[..., 2])

def inv_Part_1_By_2(x: np.ndarray):
    x = ((x >> 2) | x) & 0x030C30C3
    x = ((x >> 4) | x) & 0x0300F00F
    x = ((x >> 8) | x) & 0x030000FF
    x = ((x >>16) | x) & 0x000003FF
    return x

def inv_morton_naive(input: np.ndarray):
    x = input &        0x09249249
    y = (input >> 1) & 0x09249249
    z = (input >> 2) & 0x09249249
    
    return inv_Part_1_By_2(x), inv_Part_1_By_2(y), inv_Part_1_By_2(z)

def inv_morton(input:np.ndarray):
    x,y,z = inv_morton_naive(input)
    return np.stack([x,y,z], dim = -1)

    
def transform_msgpack(path: str, out_path: str = None):
    res = {}
    # Set File Path
    assert (os.path.isfile(path))
    dir_name, full_file_name = os.path.split(path)
    file_name, ext_name = os.path.splitext(full_file_name)
    
    if out_path is None:
        out_path = os.path.join("data", file_name)
    os.makedirs(out_path, exist_ok = True)
    
    # Load the msgpack
    with open(path, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw = False)
        config = next(unpacker)

    # Set Model Parameters
    # Total: 12206480 Parameters
    params_binary = np.frombuffer(config["snapshot"]["params_binary"], dtype = np.float16, offset = 0)
    # Transform to torch tensor
    params_binary = params_binary.astype(np.float32)
    
    # Params for Hash Encoding Network
    ## Network Params Size: 32 * 64 + 64 * 16 = 3072
    hashenc_params_network = params_binary[:(32 * 64 + 64 * 16)]
    params_binary = params_binary[(32 * 64 + 64 * 16):]
    # Params for RGB Network
    ## Network Params Size: 32 * 64 + 64 * 64 + 64 * 16 = 7168
    rgb_params_network = params_binary[:(32 * 64 + 64 * 64 + 64 * 16)]
    params_binary = params_binary[(32 * 64 + 64 * 64 + 64 * 16):]
    # Params for Hash Encoding Grid
    ## Grid size: 12196240
    hashenc_params_grid = params_binary

    
    # Occupancy Grid Part
    grid_raw = np.array(np.clip(
        np.frombuffer(config["snapshot"]["density_grid_binary"],dtype=np.float16).astype(np.float32),
        0, 1) > 0.01, dtype = np.int8)
    grid = np.zeros([128 * 128 * 128], dtype = np.int8)
    x, y, z = inv_morton_naive(np.arange(0, 128**3, 1))
    grid[x * 128 * 128 + y * 128 + z] = grid_raw
    
    # Save
    ## Hash Net Params(3072)
    np.savetxt(os.path.join(out_path, "params_3072.txt"), hashenc_params_network, fmt = "%.8f")
    ## Hash Grid Params
    np.savetxt(os.path.join(out_path, "params_hash.txt"), hashenc_params_grid, fmt = "%.8f")
    ## RGB Net
    np.savetxt(os.path.join(out_path, "params_7168.txt"), rgb_params_network, fmt = "%.8f")
    ## Occupancygrid
    np.savetxt(os.path.join(out_path, "OccupancyGrid.txt"), grid, fmt = "%d")