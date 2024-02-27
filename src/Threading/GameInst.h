#pragma once
#include "../Lists.h"
#include "../Util/AvgTracker.h"

namespace RLGPC {
	class GameInst {
	public:
		RLGSC::Gym* gym;
		RLGSC::Match* match;

		FList2 curObs;

		uint64_t totalSteps;

		float curEpRew = 0;
		AvgTracker avgStepRew, avgEpRew;

		// NOTE: Gym and match will be deleted when GameInst is deleted
		GameInst(RLGSC::Gym* gym, RLGSC::Match* match) : gym(gym), match(match) {
			totalSteps = 0;
		}

		RG_NO_COPY(GameInst);

		void ResetAvgs() {
			avgStepRew.Reset();
			avgEpRew.Reset();
		}

		void Start();
		RLGSC::Gym::StepResult Step(const IList& actions);

		~GameInst() {
			delete gym;
			delete match;
		}
	};
}