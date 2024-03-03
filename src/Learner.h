#pragma once
#include "PPO/PPOLearner.h"
#include "Threading/ThreadAgentManager.h"
#include "Util/WelfordRunningStat.h"
#include "Util/MetricSender.h"
#include "LearnerConfig.h"

namespace RLGPC {

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	class Learner {
	public:
		LearnerConfig config;

		PPOLearner* ppo;
		ThreadAgentManager* agentMgr;
		ExperienceBuffer* expBuffer;
		EnvCreateFn envCreateFn;

		int obsSize;
		int actionAmount;
		torch::Device device;

		uint64_t
			totalTimesteps = 0,
			totalEpochs = 0;
			
		WelfordRunningStat returnStats = WelfordRunningStat(1);

		Learner(EnvCreateFn envCreateFunc, LearnerConfig config);
		void Learn();
		void AddNewExperience(GameTrajectory& gameTraj);

		void Save();
		void Load();
		void SaveStats(std::filesystem::path path);
		void LoadStats(std::filesystem::path path);

		std::function<void(const Report&)> iterationCallback = NULL;

		RG_NO_COPY(Learner);

		~Learner() {
			delete ppo;
			delete agentMgr;
			delete expBuffer;
		}
	};
}