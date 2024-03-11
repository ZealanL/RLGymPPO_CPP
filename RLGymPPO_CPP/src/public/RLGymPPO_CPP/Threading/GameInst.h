#pragma once
#include "../Lists.h"
#include "../Util/AvgTracker.h"
#include "../Util/Report.h"

namespace RLGPC {
	typedef std::function<void(class GameInst*, const RLGSC::Gym::StepResult&, Report&)> StepCallback;

	// Environment creation func for each ThreadAgent
	struct EnvCreateResult {
		RLGSC::Match* match;
		RLGSC::Gym* gym;
	};
	typedef std::function<EnvCreateResult()> EnvCreateFn;

	class RG_IMEXPORT GameInst {
	public:
		RLGSC::Gym* gym;
		RLGSC::Match* match;

		FList2 curObs;

		uint64_t totalSteps;

		float curEpRew = 0;
		AvgTracker avgStepRew, avgEpRew;

		// Will be reset every iteration, when ResetMetrics() is called
		Report _metrics = {};

		StepCallback stepCallback = NULL;

		// NOTE: Gym and match will be deleted when GameInst is deleted
		GameInst(RLGSC::Gym* gym, RLGSC::Match* match) : gym(gym), match(match) {
			totalSteps = 0;
		}

		RG_NO_COPY(GameInst);

		void ResetMetrics() {
			avgStepRew.Reset();
			avgEpRew.Reset();
			_metrics.Clear();
		}

		void Start();
		RLGSC::Gym::StepResult Step(const IList& actions);

		~GameInst() {
			delete gym;
			delete match;
		}
	};
}