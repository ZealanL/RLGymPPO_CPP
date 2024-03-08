#pragma once
#include "ThreadAgent.h"
#include "../PPO/ExperienceBuffer.h"
#include <RLGymPPO_CPP/Util/Report.h>
#include <RLGymPPO_CPP/Util/WelfordRunningStat.h>
#include <RLGymPPO_CPP/Util/Timer.h>

namespace RLGPC {
	class ThreadAgentManager {
	public:
		DiscretePolicy* policy;
		std::vector<ThreadAgent*> agents;
		ExperienceBuffer* expBuffer;
		std::mutex expBufferMutex = {};
		bool standardizeOBS, autocastInference;
		uint64_t maxCollect;
		torch::Device device;

		bool disableCollection = false; // Prevents new steps from being started

		Timer iterationTimer = {};
		double lastIterationTime = 0;
		WelfordRunningStat obsStats;

		ThreadAgentManager(DiscretePolicy* policy, ExperienceBuffer* expBuffer, bool standardizeOBS, bool autocastInference, uint64_t maxCollect, torch::Device device) : 
			policy(policy), expBuffer(expBuffer), standardizeOBS(standardizeOBS), autocastInference(autocastInference), maxCollect(maxCollect), device(device) {}

		RG_NO_COPY(ThreadAgentManager);

		void CreateAgents(EnvCreateFn func, int amount, int gamesPerAgent);

		void StartAgents() {
			for (ThreadAgent* agent : agents)
				agent->Start();
		}

		void StopAgents() {
			for (ThreadAgent* agent : agents)
				agent->Stop();
		}

		void SetStepCallback(StepCallback callback) {
			for (ThreadAgent* agent : agents)
				for (GameInst* game : agent->gameInsts)
					game->stepCallback = callback;
		}

		void GetMetrics(Report& report);
		void ResetMetrics();

		GameTrajectory CollectTimesteps(uint64_t amount);

		~ThreadAgentManager() {
			for (ThreadAgent* agent : agents)
				delete agent;
		}
	};
}