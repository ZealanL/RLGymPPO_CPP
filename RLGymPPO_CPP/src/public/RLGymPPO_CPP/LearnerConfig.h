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

		bool renderMode = false;
		// If renderMode, this is the scaling of time for the game
		// 1.0 = Run the game at real time
		// 2.0 = Run the game twice as fast as real time
		float renderTimeScale = 1.5f; 

		// Set to 0 to disable
		uint64_t timestepLimit = 0;

		int64_t expBufferSize = 100 * 1000;
		int64_t timestepsPerIteration = 50 * 1000;
		bool standardizeReturns = true;
		bool standardizeOBS = false; // TODO: Implement
		int maxReturnsPerStatsInc = 150;
		int stepsPerObsStatsInc = 5;

		// Actions with the highest probability are always chosen, instead of being more likely
		// This will make your bot play better, but is horrible for learning
		// Trying to run a PPO learn iteration with deterministic mode will throw an exception
		bool deterministic = false;

		// Collect additional steps during the learning phase
		// Note that, once the learning phase completes and the policy is updated, these additional steps are from the old policy
		bool collectionDuringLearn = false;

		PPOLearnerConfig ppo = {};

		float gaeLambda = 0.95f;
		float gaeGamma = 0.99f;

		// Set to a directory with numbered subfolders, the learner will load the subfolder with the highest number
		// If the folder is empty or does not exist, loading is skipped
		// Set empty to disable loading entirely
		std::filesystem::path checkpointLoadFolder = "checkpoints"; 

		// Checkpoints are saved here as timestep-numbered subfolders
		//	e.g. a checkpoint at 20,000 steps will save to a subfolder called "20000"
		// Set empty to disable saving
		std::filesystem::path checkpointSaveFolder = "checkpoints"; 
		bool saveFolderAddUnixTimestamp = false; // Appends the unix time to checkpointSaveFolder

		// Save every timestep
		// Set to zero to just use timestepsPerIteration
		int64_t timestepsPerSave = 500 * 1000;

		int randomSeed = 123;
		int checkpointsToKeep = 5; // Checkpoint storage limit before old checkpoints are deleted, set to -1 to disable
		LearnerDeviceType deviceType = LearnerDeviceType::AUTO; // Auto will use your CUDA GPU if available

		// Send metrics to the python metrics receiver
		// The receiver can then log them to wandb or whatever
		bool sendMetrics = true;
		std::string metricsProjectName = "rlgymppo-cpp"; // Project name for the python metrics receiver
		std::string metricsGroupName = "unnamed-runs"; // Group name for the python metrics receiver
		std::string metricsRunName = "rlgymppo-cpp-run"; // Run name for the python metrics receiver
	};
}