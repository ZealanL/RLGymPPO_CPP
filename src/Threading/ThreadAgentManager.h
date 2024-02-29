#pragma once
#include "ThreadAgent.h"
#include "../PPO/ExperienceBuffer.h"
#include "../Util/Report.h"
#include "../Util/WelfordRunningStat.h"

namespace RLGPC {
	class ThreadAgentManager {
	public:
		DiscretePolicy* policy;
		std::vector<ThreadAgent*> agents;
		ExperienceBuffer* expBuffer;
		std::mutex expBufferMutex = {};
		bool standardizeOBS, autocastInference;
		torch::Device device;

		WelfordRunningStat obsStats;

		ThreadAgentManager(DiscretePolicy* policy, ExperienceBuffer* expBuffer, bool standardizeOBS, bool autocastInference, torch::Device device) : 
			policy(policy), expBuffer(expBuffer), standardizeOBS(standardizeOBS), autocastInference(autocastInference), device(device) {}

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

		void GetMetrics(Report& report);
		void ResetMetrics();

		GameTrajectory CollectTimesteps(uint64_t amount);

		~ThreadAgentManager() {
			for (ThreadAgent* agent : agents)
				delete agent;
		}
	};
}