#include "ThreadAgent.h"

#include "ThreadAgentManager.h"
#include "../Util/Timer.h"

using namespace RLGPC;

torch::Tensor MakeGamesOBSTensor(std::vector<GameInst*>& games) {
	assert(!games.empty());
	torch::Tensor obsTensor = torch::unsqueeze(FLIST2_TO_TENSOR(games[0]->curObs), 0);
	if (games.size() > 1) {
		obsTensor = obsTensor.repeat({ (int64_t)games.size(), 1, 1 });
		
		for (int i = 1; i < games.size(); i++) {
			torch::Tensor indexTensor = torch::tensor(i); // TODO: Isn't this slow and stupid
			obsTensor.index_copy_(0, indexTensor, torch::unsqueeze(FLIST2_TO_TENSOR(games[i]->curObs), 0));
		}
	}
	return obsTensor;
}

void _RunFunc(ThreadAgent* ta) {
	RG_NOGRAD;
	ta->isRunning = true;

	auto mgr = (ThreadAgentManager*)ta->_manager;
	auto& games = ta->gameInsts;
	int numGames = ta->numGames;

	auto device = mgr->device;

	// Start games
	for (auto game : games)
		game->Start();

	// Will stores our current observations for all our games
	torch::Tensor curObsTensor = MakeGamesOBSTensor(games);

	while (ta->shouldRun) {
		// Move our current OBS tensor to the device we run the policy on
		// This conversion time is not counted as a part of policy inference time
		torch::Tensor curObsTensorDevice = curObsTensor.to(device, true);

		// Infer the policy to get actions for all our agents in all our games
		Timer policyInferTimer = {};
		ta->inferenceMutex.lock();
		RG_AUTOCAST_ON();
		auto actionResults = mgr->policy->GetAction(curObsTensorDevice);
		RG_AUTOCAST_OFF();
		ta->inferenceMutex.unlock();
		ta->times.policyInferTime += policyInferTimer.Elapsed();

		// Step the gym with the actions we got
		Timer gymStepTimer = {};
		float avgRew = 0;
		auto stepResults = new RLGSC::Gym::StepResult[numGames];
		int actionsOffset = 0;
		for (int i = 0; i < numGames; i++) {
			auto game = games[i];
			int numPlayers = game->match->playerAmount;

			// Actions output has a dimension for each player, but not for each game
			// So we will need to slice the section of it that is for this game
			auto actionSlice = actionResults.action.slice(0, actionsOffset, actionsOffset + numPlayers);

			stepResults[i] = game->Step(TENSOR_TO_ILIST(actionSlice));

			actionsOffset += numPlayers;
		}

		// Make sure we got the end of actions
		// Otherwise there's a wrong number of actions for whatever reason
		assert(actionsOffset == actionResults.action.size(0));
		ta->times.envStepTime += gymStepTimer.Elapsed();

		// Update our tensor storing the next observation after the step, from each gym
		torch::Tensor nextObsTensor = MakeGamesOBSTensor(games);

		// Steps complete, add all timestep data to our trajectories, for each game
		Timer trajAppendTimer = {};
		ta->trajMutex.lock();
		for (int i = 0, actionsOffset = 0; i < numGames; i++) {
			int numPlayers = games[i]->match->playerAmount;

			auto& stepResult = stepResults[i];

			float done = (float)stepResult.done;
			float truncated = (float)false;

			auto tDone = torch::tensor(done);
			auto tTruncated = torch::tensor(truncated);
			auto states = curObsTensor[i];
			auto nextStates = nextObsTensor[i];

			for (int j = 0; j < numPlayers; j++) {
				ta->trajectories[i][j].AppendSingleStep(
					{
						states[j],
						actionResults.action[actionsOffset + j],
						actionResults.logProb[actionsOffset + j],
						torch::tensor(stepResult.reward[j]),
						nextStates[j],
						tDone,
						tTruncated
					}
				);
			}
			
			actionsOffset += numPlayers;
		}
		ta->trajMutex.unlock();
		ta->times.trajAppendTime += trajAppendTimer.Elapsed();

		// Now that the step is done, our next OBS becomes our current
		curObsTensor = nextObsTensor;

		delete[] stepResults;
	}

	ta->isRunning = false;
}

RLGPC::ThreadAgent::ThreadAgent(void* manager, int numGames, EnvCreateFn envCreateFn)
	: _manager(manager), numGames(numGames) {

	trajectories.resize(numGames);
	for (int i = 0; i < numGames; i++) {
		auto envCreateResult = envCreateFn();
		gameInsts.push_back(new GameInst(envCreateResult.gym, envCreateResult.match));
		trajectories[i].resize(envCreateResult.match->playerAmount);
	}
}

void RLGPC::ThreadAgent::Start() {
	this->shouldRun = true;
	this->thread = std::thread(_RunFunc, this);
	this->thread.detach();
}

void RLGPC::ThreadAgent::Stop() {
	this->shouldRun = false;

	// Wait for thread to stop runing
	// TODO: Lame solution
	while (isRunning)
		RG_SLEEP(1);
}