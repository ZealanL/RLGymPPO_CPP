#include "Gym.h"

namespace RLGSC {
	Gym::Gym(Match* match, int tickSkip, CarConfig carConfig, MutatorConfig mutatorConfig) :
		match(match), tickSkip(tickSkip) {
		arena = Arena::Create(GameMode::SOCCAR);
		arena->SetMutatorConfig(mutatorConfig);

		for (int i = 0; i < match->teamSize; i++) {
			arena->AddCar(Team::BLUE, carConfig);
			if (match->spawnOpponents)
				arena->AddCar(Team::ORANGE, carConfig);
		}
	}

	FList2 Gym::Reset() {
		GameState resetState = match->ResetState(arena);
		match->EpisodeReset(resetState);
		prevState = resetState;

		FList2 obs = match->BuildObservations(resetState);
		return obs;
	}

	Gym::StepResult Gym::Step(const ActionParser::Input& actionsData) {
		ActionSet actions = match->ParseActions(actionsData, prevState);
		match->prevActions = actions;

		GameState state;

		{ // Step arena with actions
			auto carItr = arena->_cars.begin();
			for (int i = 0; i < actions.size(); i++) {
				(*carItr)->controls = (CarControls)actions[i];
				carItr++;
			}

			arena->Step(1);
			state = GameState(arena);
			arena->Step(tickSkip - 1);
			totalTicks += tickSkip;
			totalSteps++;
		}

		FList2 obs = match->BuildObservations(state);
		bool done = match->IsDone(state);
		FList rewards = match->GetRewards(state, done);
		prevState = state;

		return StepResult {
			obs,
			rewards,
			done,
			state
		};
	}
}