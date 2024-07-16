#pragma once

// Include RLGymSim
#include <RLGymSim_CPP/Gym.h>

#if defined(_MSC_VER)
// MSVC
#define RG_EXPORTED __declspec(dllexport)
#define RG_IMPORTED __declspec(dllimport)
#else
// Everything else (?)
#define RG_EXPORTED __attribute__((visibility("default")))
#define RG_IMPORTED
#endif

#ifdef WITHIN_RLGPC
#define RG_IMEXPORT RG_EXPORTED
#else
#define RG_IMEXPORT RG_IMPORTED
#endif

#define RG_SLEEP(ms) std::this_thread::sleep_for(std::chrono::milliseconds(ms))

#define THREAD_WAIT() RG_SLEEP(2)