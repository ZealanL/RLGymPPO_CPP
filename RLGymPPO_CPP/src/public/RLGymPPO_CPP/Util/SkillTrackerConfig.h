#pragma once
#include "../Lists.h"

#include "../Threading/GameInst.h"

namespace RLGPC {
	struct SkillTrackerConfig {
		bool enabled = false; // Enable skill tracking

		// Env create func for eval environments
		// If NULL, the learner's env create func is used
		EnvCreateFn envCreateFunc = NULL;

		// Step callback for eval environments
		// If NULL, the learner's step callback func is used
		StepCallback stepCallback = NULL;

		int numEnvs = 1; // Number of environments for evaluation
		float simTime = 30; // Time (in seconds) to simulate each iteration
		int64_t timestepsPerVersion = 50 * 1000 * 1000; // Amout of timesteps between saving versions
		int maxVersions = 4; // Maximum amount of versions to store

		// When initialized, add the current version as the first previous version
		// If false, we will wait until timestepsPerVersion has passed to start comparing skill
		bool startWithVersion = true; 

		float ratingInc = 5; // Rating increment scale per-goal
		float initialRating = 1000; // Initial rating of the current version
	};
}