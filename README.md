# RLGymPPO_CPP
A lightning-fast C++ implementation and extension of [RLGym-PPO](https://github.com/AechPro/rlgym-ppo), as well as rlgym-sim

## Speed
Results will vary depending on hardware, but it **should be substantially faster for everyone**.

On my computer (Intel i5-11400 and GTX 3060 Ti), this repo is about 5x faster than Python RLGym-PPO on default settings.
Collection has the most substantial benefit, and I can reach upwards of 70ksps on my computer in C++, vs 10k in Python.

## Features
This implementation adds several features that RLGym-PPO/rlgym-sim doesn't have (mostly because it does not fit in Aech's scope for RLGym-PPO):
- Different multithreaded collection model that doesn't require constant cross-thread communication
- Can run far more environments (hundreds to thousands) simultaneously using far less memory per environment
- Actual multithreading used instead of separate processes and shared memory
- Fully-configurable skill tracking system using ELO
- Full RocketSim CarState/BallState access in GameState (e.g. `player.carState.isFlipping`)
- RocketSim Arena access in GameState
- Built-in zero-sum rewards with adjustable opponent scale
- Built-in padded obs builder with slot shuffling
- Support for more advanced state setters via RocketSim Arena access
- Added possibility for rewards to override their behavior across all players
- Support for collection during learn
- Support for auto-casted learn
- Better multithreading of learn for CPU-only
- Built-in gradient noise measurement system
- Added callbacks for steps and iterations
- Uses RocketSim `Vec` class with various quality-of-life functions like `.Length()`, `.Dist()`, etc.

## Accuracy to Python RLGym-PPO
According to several different learning tests, RLGymPPO_CPP and RLGym-PPO have no differences in learning.

## Installation
- Clone this repository recursively: `git clone https://github.com/ZealanL/RLGymPPO_CPP --recurse`
- If you have an NVIDIA GPU, install CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Download libtorch for CUDA 11.8 (or for CPU if you don't have an NVIDIA GPU): https://pytorch.org/get-started/locally/
- Put the `libtorch` folder inside `RLGymPPO_CPP/RLGymPPO_CPP`
- Open the main `RLGymPPO_CPP` folder as a CMake project (if you're on Windows, I recommend Visual Studio with the C++ Desktop package)
- Change the build type to `RelWithDebInfo` (`Debug` build type is very slow and not really supported) (don't worry you can still debug it)
- Make sure you have a global Python installation with `wandb` installed (unless you have turned off metrics)
- Build it
- Add your `collision_meshes` folder to wherever the executable is running

## Transferring models between C++ and Python
You can do this using the script `tools/checkpoint_converter.py`

I've confirmed that this script works perfectly, however you will need to make sure the obs builder and action parser match *perfectly* in Python

## Dependencies 
 - LibTorch (ideally with CUDA support)
 - https://github.com/ZealanL/RLGymSim_CPP (already included)
 - https://github.com/nlohmann/json (already included)
