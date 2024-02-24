#pragma once
#include "ThreadAgent.h"
#include "../PPO/ExperienceBuffer.h"
#include "../Util/WelfordRunningStat.h"

namespace RLGPC {
	class ThreadAgentManager {
	public:
		DiscretePolicy* policy;
		std::vector<ThreadAgent*> agents;
		ExperienceBuffer* expBuffer;
		std::mutex expBufferMutex = {};
		bool standardizeOBS;
		torch::Device device;

		WelfordRunningStat obsStats;

		ThreadAgentManager(DiscretePolicy* policy, ExperienceBuffer* expBuffer, bool standardizeOBS, torch::Device device) : 
			policy(policy), expBuffer(expBuffer), standardizeOBS(standardizeOBS), device(device) {}

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

		ThreadAgent::Times GetTotalAgentTimes();

		void ResetAgentTimes() {
			for (ThreadAgent* agent : agents) {
				agent->times = {};
			}
		}

		float GetAvgReward() {
			float avg = 0;
			for (ThreadAgent* agent : agents)
				avg += agent->avgRew;
			return avg / agents.size();
		}

		void ResetAvgReward() {
			for (ThreadAgent* agent : agents)
				agent->ResetAvgReward();
		}

		GameTrajectory CollectTimesteps(uint64_t amount);

		~ThreadAgentManager() {
			for (ThreadAgent* agent : agents)
				delete agent;
		}
	};
}