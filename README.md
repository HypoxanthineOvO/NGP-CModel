# NGP C-Model
## Introduction
This is a C Model of instant-ngp. All modules are implemented by C++, only with `Eigen`.

It's more convinient for translate this C Model to RTL code; besides, if you are interest in NGP's implementation, you can read this code.

> Now, only an Lego model is supported. I will quickly inplement `.msgpack` loading
> The test data is too big. If you need it, please contact me.

## Quick Start
### Install XMake
[Xmake](https://xmake.io/#/getting_started)

### Get the data
- Edit `transform.py`'s main part: `load_msgpack({YOUR PATH})`, then you will find data in `data/{NAME}/`
- Then, edit the `main.cpp`'s `NAME` variable to your data name.

### Build and Run this Project
Run `xmake` to build the project. You may need install some package by input `y`.

Then, Run `./Instant-NGP` to run the project.

## Quality (PSNR)
> Evaluate in NeRF-Synthetic Dataset
- Avg: 34.33
  - chair: 35.76
  - drums: 28.89
  - ficus: 33.97
  - hotdog: 39.00
  - lego: 33.16
  - materials: 33.14
  - mic: 37.73
  - ship: 32.97