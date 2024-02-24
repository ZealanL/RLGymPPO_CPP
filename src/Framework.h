#pragma once

// Include RLGymSim
#include "../RLGymSim_CPP/src/Gym.h"

// Include torch
// TODO: Don't make everyone that uses this library have to include torch, because dear god
#include <torch/all.h>

#define RG_SLEEP(ms) std::this_thread::sleep_for(std::chrono::milliseconds(ms))

#define RG_NOGRAD torch::NoGradGuard _noGradGuard

// Print a number with commas
static inline std::string _RG_COMMA_INT(int64_t i) {
	std::string s = std::to_string(i);
	int n = s.length() - 3;
	int end = (i >= 0) ? 0 : 1;
	while (n > end) {
		s.insert(n, ",");
		n -= 3;
	}
	return s;
}
#define RG_COMMA_INT(i) _RG_COMMA_INT(i) 