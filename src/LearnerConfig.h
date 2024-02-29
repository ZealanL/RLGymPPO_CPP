#pragma once
#include "Lists.h"
#include "PPO/PPOLearnerConfig.h"

namespace RLGPC {
	enum class LearnerDeviceType {
		AUTO,
		CPU,
		GPU_CUDA
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	struct LearnerConfig {
		int numThreads = 8;
		int numGamesPerThread = 16;
		int minInferenceSize = 80;
		bool render = false;
		int renderDelayMS = 0;

		// Set to 0 to disable
		uint64_t timestepLimit = 0;

		int expBufferSize = 100 * 1000;
		int timestepsPerIteration = 50 * 1000;
		bool standardizeReturns = true;
		bool standardizeOBS = false; // TODO: Implement
		int maxReturnsPerStatsInc = 150;
		int stepsPerObsStatsInc = 5;

		bool autocastInference = true; // Enable torch autocast for policy inference

		PPOLearnerConfig ppo = {};

		float gaeLambda = 0.95f;
		float gaeGamma = 0.99f;

		std::filesystem::path checkpointLoadFolder = {}; // Set empty to disable
		std::filesystem::path checkpointSaveFolder = "checkpoints"; // Set empty to disable
		bool saveFolderAddUnixTimestamp = true;
		int saveEveryTS = 1000 * 1000; // Save every timestep

		int randomSeed = 123;
		int checkpointsToKeep = 5; // Checkpoint storage limit before old checkpoints are deleted, set to -1 to disable
		int shmBufferSize = 8 * 1024;
		LearnerDeviceType deviceType = LearnerDeviceType::AUTO; // Auto will use your CUDA GPU if available
	};
}