#include "ThreadAgent.h"

#include "ThreadAgentManager.h"
#include <RLGymPPO_CPP/Util/Timer.h>

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

	bool render = mgr->renderSender;

	Timer stepTimer = {};

	// Start games
	for (auto game : games)
		game->Start();

	// Will stores our current observations for all our games
	torch::Tensor curObsTensor = MakeGamesOBSTensor(games);

#if 0 // TODO: Potential cause of learning errors
	bool halfPrec = mgr->policyHalf != NULL;
#else
	constexpr bool halfPrec = false;
#endif
	while (ta->shouldRun) {

		if (render)
			stepTimer.Reset();

		// Don't run if we reached our step limit
		while (ta->stepsCollected > ta->maxCollect)
			std::this_thread::yield();

		while (mgr->disableCollection)
			std::this_thread::yield();

		// Move our current OBS tensor to the device we run the policy on
		// This conversion time is not counted as a part of policy inference time
		torch::Tensor curObsTensorDevice;
		if (halfPrec) {
			curObsTensorDevice = curObsTensor.to(RG_HALFPERC_TYPE).to(device, true);
		} else {
			curObsTensorDevice = curObsTensor.to(device, true);
		}

		// Infer the policy to get actions for all our agents in all our games
		Timer policyInferTimer = {};
		
		auto actionResults = (halfPrec ? mgr->policyHalf : mgr->policy)->GetAction(curObsTensorDevice);
		if (halfPrec) {
			actionResults.action = actionResults.action.to(torch::ScalarType::Float);
			actionResults.logProb = actionResults.logProb.to(torch::ScalarType::Float);
		}

		float policyInferTime = policyInferTimer.Elapsed();
		ta->times.policyInferTime += policyInferTime;

		// Step the gym with the actions we got
		Timer gymStepTimer = {};
		ta->gameStepMutex.lock();
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
		ta->gameStepMutex.unlock();

		// Make sure we got the end of actions
		// Otherwise there's a wrong number of actions for whatever reason
		assert(actionsOffset == actionResults.action.size(0));
		float envStepTime = gymStepTimer.Elapsed();
		ta->times.envStepTime += envStepTime;

		// Update our tensor storing the next observation after the step, from each gym
		torch::Tensor nextObsTensor = MakeGamesOBSTensor(games);

		if (!render) {
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

				ta->stepsCollected += numPlayers;
				actionsOffset += numPlayers;
			}
			ta->trajMutex.unlock();
			ta->times.trajAppendTime += trajAppendTimer.Elapsed();
		} else {
			// Update renderer
			auto renderSender = mgr->renderSender;
			auto renderGame = games[0];
			renderSender->Send(renderGame->gym->prevState, RLGSC::ActionSet());

			// Delay for render
			// TODO: Somewhat dumb system using static variables
			static auto lastRenderTime = std::chrono::high_resolution_clock::now();
			auto durationSince = std::chrono::high_resolution_clock::now() - lastRenderTime;
			lastRenderTime = std::chrono::high_resolution_clock::now();

			int64_t micsSince = std::chrono::duration_cast<std::chrono::microseconds>(durationSince).count();

			double timeTaken = stepTimer.Elapsed();
			double targetTime = (1/120.0) * renderGame->gym->tickSkip * mgr->renderTimeScale;
			double sleepTime = RS_MAX(targetTime - timeTaken, 0);
			int64_t sleepMics = (int64_t)(sleepTime * 1000.0 * 1000.0);

			std::this_thread::sleep_for(std::chrono::microseconds(sleepMics));
		}

		// Now that the step is done, our next OBS becomes our current
		curObsTensor = nextObsTensor;

		delete[] stepResults;
	}

	ta->isRunning = false;
}

RLGPC::ThreadAgent::ThreadAgent(void* manager, int numGames, uint64_t maxCollect, EnvCreateFn envCreateFn)
	: _manager(manager), numGames(numGames), maxCollect(maxCollect) {

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