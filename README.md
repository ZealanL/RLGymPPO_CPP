# RLGymPPO_CPP
A lightning-fast C++ implementation of [RLGym-PPO](https://github.com/AechPro/rlgym-ppo)

## Speed
Results will vary depending on hardware, but it **should be substantially faster for everyone**.

On my computer (Intel i5-11400 and GTX 3060 Ti), this repo is about 3x faster than Python RLGym-PPO.

## Accuracy to Python RLGym-PPO
According to a few different learning tests, RLGymPPO_CPP and RLGym-PPO have no differences in learning.

Most of these tests involved training on simple and complex rewards for up to 10m steps,
and more tests should probably be ran for much longer training sessions to confirm.

## Installation
- Clone this repository recursively: `git clone https://github.com/ZealanL/RLGymPPO_CPP --recurse`
- If you have an NVIDIA GPU, install CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Download libtorch for CUDA 11.8 (or for CPU if you don't have an NVIDIA GPU): https://pytorch.org/get-started/locally/
- Put the `libtorch` folder inside `RLGymPPO_CPP/RLGymPPO_CPP`
- Open the main `RLGymPPO_CPP` folder as a CMake project (if you're on Windows, I recommend Visual Studio with the C++ Desktop package)
- Change the build type to `RelWithDebInfo` (`Debug` build type is very slow and not really supported) (don't worry you can still debug it)
- Make sure you have a global Python installation with `wandb` installed (unless you have turned off metrics)
- Build it!

## Transferring models between C++ and Python
You can do this using the script `tools/checkpoint_converter.py`

I've confirmed that this script works perfectly, however you will need to make sure the obs builder and action parser match *perfectly* in Python

## Dependencies 
 - LibTorch (ideally with CUDA support)
    - Download from https://pytorch.org/get-started/locally/ by selecting "C++/Java" as your language
    - Place the 'libtorch' folder within the 'RLGymPPO_CPP' folder
    - Pray
 - https://github.com/ZealanL/RLGymSim_CPP (already included)
 - https://github.com/nlohmann/json (already included)
