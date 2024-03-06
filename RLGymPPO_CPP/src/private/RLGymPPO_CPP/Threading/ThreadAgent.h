#pragma once
#include "../PPO/DiscretePolicy.h"
#include <RLGymPPO_CPP/Threading/GameInst.h>
#include "GameTrajectory.h"

namespace RLGPC {
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
		std::atomic<uint64_t> stepsCollected = 0;
		uint64_t maxCollect;
		
		// Lock to prevent game stepping
		std::mutex gameStepMutex = {};

		// Lock to modify trajectories
		std::mutex trajMutex = {};

		ThreadAgent(void* manager, int numGames, uint64_t maxCollect, EnvCreateFn envCreateFn);

		RG_NO_COPY(ThreadAgent);

		void Start();
		void Stop();

		~ThreadAgent() {
			for (auto g : gameInsts)
				delete g;
		}
	};
}