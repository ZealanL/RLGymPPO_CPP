#pragma once
#include "../Lists.h"

namespace RLGPC {
	class GameInst {
	public:
		RLGSC::Gym* gym;
		RLGSC::Match* match;

		FList2 curObs;

		uint64_t totalSteps;

		float totalRew = 0;
		uint64_t totalRewCount = 0;

		// NOTE: Gym and match will be deleted when GameInst is deleted
		GameInst(RLGSC::Gym* gym, RLGSC::Match* match) : gym(gym), match(match) {
			totalSteps = 0;
		}

		RG_NO_COPY(GameInst);

		float GetAvgReward() {
			if (totalRewCount > 0) {
				return totalRew / totalRewCount;
			} else {
				return NAN;
			}
		}

		void ResetAvgReward() {
			totalRew = totalRewCount = 0;
		}

		void Start();
		RLGSC::Gym::StepResult Step(const IList& actions);

		~GameInst() {
			delete gym;
			delete match;
		}

	};
}