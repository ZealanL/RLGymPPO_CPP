# RLGymPPO_CPP
A lightning-fast C++ implementation of [RLGym-PPO](https://github.com/AechPro/rlgym-ppo)

## WORK IN PROGRESS 
This library is still a work in progress and is missing some features that RLGym-PPO has.

## Speed
Results will vary depending on hardware, but it **should be substantially faster for everyone**.

On my computer (Intel i5-11400 and GTX 3060 Ti), this repo is about 3x faster than Python RLGym-PPO.

## Accuracy to Python RLGym-PPO
According to a few different learning tests, RLGymPPO_CPP and RLGym-PPO have no differences in learning.

Most of these tests involved training on simple and complex rewards for up to 10m steps,
and more tests should probably be ran for much longer training sessions to confirm.

## Transferring models between C++ and Python
You can do this using the script `tools/checkpoint_converter.py`

This script is very new, so if you find any bugs, let me know

## Dependencies 
 - LibTorch (ideally with CUDA support)
    - Download from https://pytorch.org/get-started/locally/ by selecting "C++/Java" as your language
    - Place the 'libtorch' folder within the 'RLGymPPO_CPP' folder
    - Pray
 - https://github.com/ZealanL/RLGymSim_CPP (already included)
 - https://github.com/nlohmann/json (already included)