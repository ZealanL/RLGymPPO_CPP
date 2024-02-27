#pragma once
#include "../PPO/DiscretePolicy.h"
#include "GameInst.h"
#include "GameTrajectory.h"

namespace RLGPC {
	// Environment creation func for each ThreadAgent
	struct EnvCreateResult {
		RLGSC::Match* match;
		RLGSC::Gym* gym;
	};
	typedef std::function<EnvCreateResult()> EnvCreateFn;

	class ThreadAgent {
	public:

		void* _manager;
		std::thread thread;

		int numGames;
		std::vector<GameInst*> gameInsts;

		bool shouldRun = false; // Set from thread
		std::atomic<bool> isRunning = false;

		struct Times {
			double
				envStepTime = 0,
				policyInferTime = 0,
				trajAppendTime = 0;

			double* begin() {
				return &envStepTime;
			}

			double* end() {
				return &trajAppendTime + 1;
			}
		};
		Times times = {}; // TODO: Convert to use Report instead

		std::vector<std::vector<GameTrajectory>> trajectories = {};
		std::mutex trajMutex = {};

		ThreadAgent(void* manager, int numGames, EnvCreateFn envCreateFn);

		RG_NO_COPY(ThreadAgent);

		void Start();
		void Stop();

		~ThreadAgent() {
			for (auto g : gameInsts)
				delete g;
		}
	};
}