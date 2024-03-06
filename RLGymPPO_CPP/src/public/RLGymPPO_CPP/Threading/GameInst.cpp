#include "GameInst.h"

void RLGPC::GameInst::Start() {
	curObs = gym->Reset();
}

RLGSC::Gym::StepResult RLGPC::GameInst::Step(const IList& actions) {

	// Step with agent actions
	auto stepResult = gym->Step(actions);

	auto& nextObs = stepResult.obs;

	{ // Update avg rewards
		float totalRew = 0;
		for (int i = 0; i < match->playerAmount; i++)
			totalRew += stepResult.reward[i];

		avgStepRew.Add(totalRew, match->playerAmount);
		curEpRew += totalRew / match->playerAmount;
	}

	if (stepCallback)
		stepCallback(this, stepResult, _metrics);

	// Environment ending
	if (stepResult.done) {
		nextObs = gym->Reset();
		
		avgEpRew += curEpRew;
		curEpRew = 0;
	}

	curObs = nextObs;
	totalSteps++;

	return stepResult;
}