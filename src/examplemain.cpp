#include "Learner.h"

#include "../RLGymSim_CPP/src/Utils/RewardFunctions/CommonRewards.h"
#include "../RLGymSim_CPP/src/Utils/TerminalConditions/NoTouchCondition.h"
#include "../RLGymSim_CPP/src/Utils/TerminalConditions/GoalScoreCondition.h"
#include "../RLGymSim_CPP/src/Utils/OBSBuilders/DefaultOBS.h"
#include "../RLGymSim_CPP/src/Utils/StateSetters/RandomState.h"
#include "../RLGymSim_CPP/src/Utils/ActionParsers/DiscreteAction.h"

using namespace RLGPC; // RLGymPPO
using namespace RLGSC; // RLGymSim

// Create the RLGymSim environment for each of our games
EnvCreateResult EnvCreateFunc() {
	constexpr int TICK_SKIP = 8;
	constexpr float NO_TOUCH_TIMEOUT_SECS = 3.f;

	auto reward = new FaceBallReward();

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(NO_TOUCH_TIMEOUT_SECS * 120 / TICK_SKIP),
		new GoalScoreCondition()
	};

	auto obs = new DefaultOBS();
	auto actionParser = new DiscreteAction();
	auto stateSetter = new RandomState(true, true, true);

	Match* match = new Match(
		reward,
		terminalConditions,
		obs,
		actionParser,
		stateSetter,

		1, // Team size
		true // Spawn opponents
	);

	Gym* gym = new Gym(match, TICK_SKIP);
	return { match, gym };
}

int main() {
	// Initialize RocketSim with collision meshes
	RocketSim::Init("./collision_meshes");

	// Make configuration for the learner
	LearnerConfig cfg = {};

	// Play around with these to see what the optimal is for your machine
	// I personally find having less threads than my CPU thread count actually yields more SPS
	cfg.numThreads = 8;
	cfg.numGamesPerThread = 16;

	// We want a large itr/batch size
	// Increase this up to ~700k, stop if you start losing steps/second
	int tsPerItr = 100 * 1000;
	cfg.timestepsPerIteration = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 100 * 1000; // Lower this if too much VRAM is being allocated
	cfg.expBufferSize = tsPerItr * 3;
	cfg.ppo.epochs = 1;

	// Reasonable starting entropy
	cfg.ppo.entCoef = 0.005f;

	// Decently-strong learning rate to start, may start to be too high around 50m-100m steps
	cfg.ppo.policyLR = 2e-3;
	cfg.ppo.criticLR = 2e-3;

	// Layer sizes that are the default size
	cfg.ppo.policyLayerSizes = { 512, 512, 512 };
	cfg.ppo.criticLayerSizes = { 512, 512, 512 };

	// Make the learner with the environment creation function and the config we just made
	Learner learner = Learner(EnvCreateFunc, cfg);

	// Start learning!
	learner.Learn();

	return 0;
}