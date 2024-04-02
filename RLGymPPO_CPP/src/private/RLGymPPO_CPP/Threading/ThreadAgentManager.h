#pragma once
#include "ThreadAgent.h"
#include "../PPO/ExperienceBuffer.h"
#include <RLGymPPO_CPP/Util/Report.h>
#include <RLGymPPO_CPP/Util/WelfordRunningStat.h>
#include <RLGymPPO_CPP/Util/Timer.h>
#include <RLGymPPO_CPP/Util/RenderSender.h>

namespace RLGPC {
	class ThreadAgentManager {
	public:
		DiscretePolicy* policy, *policyHalf;
		std::vector<ThreadAgent*> agents;
		ExperienceBuffer* expBuffer;
		std::mutex expBufferMutex = {};
		bool standardizeOBS;
		bool deterministic;
		uint64_t maxCollect;
		torch::Device device;

		RenderSender* renderSender = NULL;
		float renderTimeScale = 1.f;

		bool disableCollection = false; // Prevents new steps from being started

		Timer iterationTimer = {};
		double lastIterationTime = 0;
		WelfordRunningStat obsStats;

		ThreadAgentManager(
			DiscretePolicy* policy, DiscretePolicy* policyHalf, ExperienceBuffer* expBuffer, 
			bool standardizeOBS, bool deterministic, uint64_t maxCollect, torch::Device device) : 
			policy(policy), policyHalf(policyHalf), expBuffer(expBuffer), 
			standardizeOBS(standardizeOBS), deterministic(deterministic), maxCollect(maxCollect), device(device) {}

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