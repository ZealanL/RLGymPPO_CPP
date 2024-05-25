#include "ThreadAgent.h"

#include "ThreadAgentManager.h"
#include <RLGymPPO_CPP/Util/Timer.h>

using namespace RLGPC;

torch::Tensor MakeGamesOBSTensor(std::vector<GameInst*>& games) {
	// TODO: Use of torch::concat is likely slow

	assert(!games.empty());
	std::vector<torch::Tensor> obsTensors = {};
	for (auto game : games)
		obsTensors.push_back(FLIST2_TO_TENSOR(game->curObs));

	try {
		return torch::concat(obsTensors);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Failed to concat OBS tensors: " << e.what());
		return {};
	}
}

void _RunFunc(ThreadAgent* ta) {
	RG_NOGRAD;
	ta->isRunning = true;

	auto mgr = (ThreadAgentManager*)ta->_manager;
	auto& games = ta->gameInsts;
	int numGames = ta->numGames;

	auto device = mgr->device;

	bool render = mgr->renderSender != NULL;
	if (render && mgr->renderDuringTraining) {
		if (ta->index != 0)
			render = false;
	}
	bool deterministic = mgr->deterministic;

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

	auto policy = (halfPrec ? mgr->policyHalf : mgr->policy);

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
		
		auto actionResults = policy->GetAction(curObsTensorDevice, deterministic);
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
			for (int i = 0, playerOffset = 0; i < numGames; i++) {
				int numPlayers = games[i]->match->playerAmount;

				auto& stepResult = stepResults[i];

				float done = (float)stepResult.done;
				float truncated = (float)false;

				auto tDone = torch::tensor(done);
				auto tTruncated = torch::tensor(truncated);

				for (int j = 0; j < numPlayers; j++) {
					ta->trajectories[i][j].AppendSingleStep(
						{
							curObsTensor[playerOffset + j],
							actionResults.action[playerOffset + j],
							actionResults.logProb[playerOffset + j],
							torch::tensor(stepResult.reward[j]),

#ifdef RG_PARANOID_MODE
							torch::Tensor(),
#endif

							nextObsTensor[playerOffset + j],
							tDone,
							tTruncated
						}
					);
				}

				ta->stepsCollected += numPlayers;
				playerOffset += numPlayers;
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
			{
				namespace chr = std::chrono;
				static auto lastRenderTime = chr::high_resolution_clock::now();
				auto durationSince = chr::high_resolution_clock::now() - lastRenderTime;
				lastRenderTime = chr::high_resolution_clock::now();

				int64_t micsSince = chr::duration_cast<chr::microseconds>(durationSince).count();

				double timeTaken = stepTimer.Elapsed();
				double targetTime = (1 / 120.0) * renderGame->gym->tickSkip / mgr->renderTimeScale;
				double sleepTime = RS_MAX(targetTime - timeTaken, 0);
				int64_t sleepMics = (int64_t)(sleepTime * 1000.0 * 1000.0);

				std::this_thread::sleep_for(chr::microseconds(sleepMics));
			}
		}

		// Now that the step is done, our next OBS becomes our current
		curObsTensor = nextObsTensor;

		delete[] stepResults;
	}

	ta->isRunning = false;
}

RLGPC::ThreadAgent::ThreadAgent(void* manager, int numGames, uint64_t maxCollect, EnvCreateFn envCreateFn, int index)
	: _manager(manager), numGames(numGames), maxCollect(maxCollect), index(index) {

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