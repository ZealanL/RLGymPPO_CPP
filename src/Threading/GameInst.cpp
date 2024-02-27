#include "GameInst.h"

void RLGPC::GameInst::Start() {
	curObs = gym->Reset();
}

RLGSC::Gym::StepResult RLGPC::GameInst::Step(const IList& actions) {

	// Step with agent actions
	auto stepData = gym->Step(actions);

	auto& nextObs = stepData.obs;

	{ // Update avg rewards
		float totalRew = 0;
		for (int i = 0; i < match->playerAmount; i++)
			totalRew += stepData.reward[i];

		avgStepRew.Add(totalRew, match->playerAmount);
		curEpRew += totalRew / match->playerAmount;
	}

	// Environment ending
	if (stepData.done) {
		nextObs = gym->Reset();
		
		avgEpRew += curEpRew;
		curEpRew = 0;
	}

	curObs = nextObs;
	totalSteps++;

	return stepData;
}