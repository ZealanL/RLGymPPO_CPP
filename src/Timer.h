#pragma once
#include "Framework.h"

namespace RLGPC {
	struct Timer {
		std::chrono::steady_clock::time_point startTime;

		Timer() {
			Reset();
		}

		// Returns elapsed time in seconds
		double Elapsed() {
			auto endTime = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = endTime - startTime;
			return elapsed.count();
		}

		void Reset() {
			startTime = std::chrono::high_resolution_clock::now();
		}
	};
}