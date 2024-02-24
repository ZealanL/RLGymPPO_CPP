#include "GameInst.h"

void RLGPC::GameInst::Start() {
	curObs = gym->Reset();
}

RLGSC::Gym::StepResult RLGPC::GameInst::Step(const IList& actions) {

	// Step with agent actions
	auto stepData = gym->Step(actions);

	auto& nextObs = stepData.obs;

	{ // Update avg reward
		for (int i = 0; i < match->playerAmount; i++)
			totalRew += stepData.reward[i];
		totalRewCount += match->playerAmount;
	}

	// Environment ending
	if (stepData.done)
		nextObs = gym->Reset();

	curObs = nextObs;
	totalSteps++;

	return stepData;
}