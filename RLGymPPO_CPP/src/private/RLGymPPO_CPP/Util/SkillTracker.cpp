#include "SkillTracker.h"
#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Utils/StateSetters/KickoffState.h>
#include <RLGymSim_CPP/Math.h>

#include <vector>

using namespace RLGSC;
using namespace RLGPC;

class DummyReward : public RLGSC::RewardFunction {
	virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
		return 0;
	}
};

static thread_local DummyReward* g_DummyReward = new DummyReward();

std::string ModeNameFromGameInst(RLGPC::GameInst* gameInst) {
	if (gameInst->match->spawnOpponents) {
		return RS_STR(gameInst->match->teamSize << "v" << gameInst->match->teamSize);
	} else {
		return RS_STR(gameInst->match->teamSize << "v0");
	}
}

RLGPC::SkillTracker::SkillTracker(const SkillTrackerConfig& config, RenderSender* renderSender) 
	: config(config), renderSender(renderSender) {

	RG_ASSERT(config.numThreads > 0);
	RG_ASSERT(config.numEnvs > 0);
	RG_ASSERT(config.timestepsPerVersion >= 0);
	RG_ASSERT(config.maxVersions > 0);
	RG_ASSERT(config.simTime > 0);

	curRating = {};

	if (!config.perModeRatings)
		curRating.data[""] = config.initialRating;

	for (int i = 0; i < config.numEnvs; i++) {

		RG_ASSERT(config.envCreateFunc);

		auto envCreateResult = config.envCreateFunc();
		GameInst* gameInst = new GameInst(envCreateResult.gym, envCreateResult.match);
		gameInst->isEval = true;
		if (config.kickoffStatesOnly)
			gameInst->match->stateSetter = new RLGSC::KickoffState();

		gameInst->match->rewardFn = g_DummyReward;
		gameInst->Start();

		if (config.perModeRatings) {
			std::string modeName = ModeNameFromGameInst(gameInst);
			modeNames.insert(modeName);
			curRating.data[modeName] = config.initialRating;
		}

		games.push_back(Game(gameInst, 1));
	}
}

RLGPC::SkillTracker::~SkillTracker() {
	for (auto& policy : oldPolicies)
		delete policy;

	for (auto game : games)
		delete game.gameInst;
}

void RLGPC::SkillTracker::UpdateRatings(RatingSet& winner, RatingSet& loser, bool updateWinner, bool updateLoser, std::string mode) {
	// Simple elo calculation

	RG_ASSERT(winner.data.contains(mode));
	RG_ASSERT(loser.data.contains(mode));

	float expDelta = (loser.data[mode] - winner.data[mode]) / 400;
	float expected = 1 / (powf(10, expDelta) + 1);

	if (updateWinner)
		winner.data[mode] += config.ratingInc * (1 - expected);
	if (updateLoser)
		loser.data[mode] += config.ratingInc * (expected - 1);
}

void RunThread(
	RLGPC::SkillTracker* self, std::vector<RLGPC::SkillTracker::Game*> games, DiscretePolicy* curPolicy, float timePerGame, 
	int threadIdx, std::mutex* ratingMutex, char* done) {
	constexpr const char* ERR_PREFIX = "RLGPC::SkillTracker RunThread(): ";
	
	for (auto gamePtr : games) {
		auto& game = *gamePtr;
		auto& gameInst = game.gameInst;
		gameInst->stepCallback = self->config.stepCallback;

		DiscretePolicy* oldPolicy = self->oldPolicies[game.oldPolicyIndex];
		int tickSkip = gameInst->gym->tickSkip;
		int numSteps = timePerGame * 120 / tickSkip;
		if (numSteps <= 0)
			RG_ERR_CLOSE(ERR_PREFIX << "simTime is too low for the number of games, there is not enough time per game to step");

		for (int i = 0; i < numSteps; i++) {
			RG_NOGRAD;

			FList2 curObsSet = gameInst->curObs;

			FList2 teamObsSets[2] = {};
			for (int j = 0; j < gameInst->match->playerAmount; j++)
				teamObsSets[(int)gameInst->gym->prevState.players[j].team].push_back(curObsSet[j]);

			auto bluePolicy = game.teamSwap ? oldPolicy : curPolicy;
			auto orangePolicy = game.teamSwap ? curPolicy : oldPolicy;

			auto blueObs = FLIST2_TO_TENSOR(teamObsSets[0]).to(bluePolicy->device);
			auto orangeObs = FLIST2_TO_TENSOR(teamObsSets[1]).to(orangePolicy->device);

			auto blueActions = TENSOR_TO_ILIST(bluePolicy->GetAction(blueObs, 1).action);
			auto orangeActions = TENSOR_TO_ILIST(orangePolicy->GetAction(orangeObs, 1).action);

			IList allActions = {};
			for (int j = 0, blueIdx = 0, orangeIdx = 0; j < gameInst->match->playerAmount; j++) {
				Team playerTeam = gameInst->gym->prevState.players[j].team;
				if (playerTeam == Team::BLUE) {
					allActions.push_back(blueActions[blueIdx]);
					blueIdx++;
				} else {
					allActions.push_back(orangeActions[orangeIdx]);
					orangeIdx++;
				}
			}

			auto stepResult = gameInst->Step(allActions);
			if (RLGSC::Math::IsBallScored(stepResult.state.ball.pos)) {
				auto scoringPolicy = (stepResult.state.ball.pos.y > 0) ? bluePolicy : orangePolicy;
				std::string modeName = ModeNameFromGameInst(game.gameInst);
				if (!self->config.perModeRatings)
					modeName = "";

				ratingMutex->lock();
				if (scoringPolicy == curPolicy) {
					// Current policy scored
					self->UpdateRatings(self->curRating, self->oldRatings[game.oldPolicyIndex], true, false, modeName);
				} else {
					// Old policy scored
					self->UpdateRatings(self->oldRatings[game.oldPolicyIndex], self->curRating, false, true, modeName);
				}
				ratingMutex->unlock();
			}

			if (stepResult.done)
				game.Reset(self->oldPolicies.size());

			if (self->renderSender && threadIdx == 0) {
				self->renderSender->Send(stepResult.state, gameInst->match->prevActions);
				float sleepTime = gameInst->gym->tickSkip / 120.f;
				std::this_thread::sleep_for(std::chrono::microseconds(int64_t(sleepTime * 1000 * 1000)));
			}
		}
	}

	*done = true;
}

void RLGPC::SkillTracker::RunGames(DiscretePolicy* curPolicy, int64_t timestepsDelta) {
	constexpr const char* ERR_PREFIX = "RLGPC::SkillTracker::RunGames(): ";

	if (runCounter % config.updateInterval != 0) {
		runCounter++;
		return;
	} else {
		runCounter++;
	}

	if (oldPolicies.empty() && config.startWithVersion) {
		DiscretePolicy* newOldPolicy = new DiscretePolicy(curPolicy->inputAmount, curPolicy->actionAmount, curPolicy->layerSizes, curPolicy->device);
		curPolicy->CopyTo(*newOldPolicy);
		oldPolicies.push_back(newOldPolicy);
		oldRatings.push_back(curRating);
	}

	if (oldPolicies.size() > 0) {
		RatingSet prevRating = curRating;

		float timePerGame = config.simTime / games.size();

		std::vector<std::thread*> threads = {};
		std::vector<char> threadDones = {};
		std::vector<std::vector<Game*>> threadGameSets = {};
		threads.resize(config.numThreads);
		threadDones.resize(config.numThreads);
		threadGameSets.resize(config.numThreads);
		for (int gameIdx = 0; gameIdx < games.size(); gameIdx++)
			threadGameSets[gameIdx % config.numThreads].push_back(&games[gameIdx]);

		std::mutex ratingMutex = {};

		for (int i = 0; i < config.numThreads; i++) {
			if (!threadGameSets[i].empty()) {
				std::thread* thread = new std::thread(RunThread, this, threadGameSets[i], curPolicy, timePerGame, i, &ratingMutex, &threadDones[i]);
				thread->detach();
				threads[i] = thread;
			} else {
				threads[i] = NULL;
			}
		}

		for (int i = 0; i < threads.size(); i++) {
			if (threads[i] == NULL)
				continue;

			while (!threadDones[i])
				std::this_thread::yield();

			if (threads[i]->joinable())
				threads[i]->join();
			delete threads[i];
		}
		threads.clear();

		RG_LOG("New ratings:");
		for (auto& pair : curRating.data) {
			if (prevRating.data.find(pair.first) == prevRating.data.end())
				continue;

			float prev = prevRating.data[pair.first];
			float delta = pair.second - prev;
			RG_LOG(
				" > " << pair.first << (pair.first.empty() ? "" : " ") << std::setprecision(6) << pair.second << 
				" (" << (delta >= 0 ? "+" : "") << std::setprecision(4) << delta << ")"
			);
		}
	} else {
		RG_LOG(" > No old policies yet, skipping");
	}

	timestepsSinceVersionMade += timestepsDelta;
	if (timestepsSinceVersionMade >= config.timestepsPerVersion) {
		// Reset all games
		for (auto game : games)
			game.gameInst->Start();

		timestepsSinceVersionMade = 0;

		// Add current policy as previous version
		DiscretePolicy* newOldPolicy = new DiscretePolicy(curPolicy->inputAmount, curPolicy->actionAmount, curPolicy->layerSizes, curPolicy->device);
		curPolicy->CopyTo(*newOldPolicy);
		oldPolicies.push_back(newOldPolicy);
		oldRatings.push_back(curRating);

		if (oldPolicies.size() > config.maxVersions) {
			delete oldPolicies[0];
			oldPolicies.erase(oldPolicies.begin());
			oldRatings.erase(oldRatings.begin());
		}
	}
}

RLGPC::SkillTracker::RatingSet RLGPC::SkillTracker::LoadRatingSet(const nlohmann::json& json, bool warn) {
	constexpr const char* ERR_PREFIX = "RLGPC::SkillTracker::LoadRatingSet(): ";

	RatingSet result = {};
	if (json.is_object()) {
		if (config.perModeRatings) {
			for (auto& modeName : modeNames) {
				if (!json.contains(modeName)) {
					RG_LOG(ERR_PREFIX << "Loaded ratings are missing mode \"" << modeName << "\", rating will be set to initial rating.");
					result.data[modeName] = config.initialRating;
				}
			}

			for (auto& pair : json.items())
				result.data[pair.key()] = pair.value();

		} else {
			RG_LOG(ERR_PREFIX << "Loaded ratings are per-mode, but per-mode ratings are disabled. Rating will be set to initial rating.");
			result.data[""] = config.initialRating;
		}
	} else {
		RG_ASSERT(json.is_number());

		if (config.perModeRatings) {
			RG_LOG(ERR_PREFIX << "Loaded ratings are not per-mode, all mode ratings will be set to the loaded rating: " << (float)json);
			for (auto& modeName : modeNames)
				result.data[modeName] = (float)json;
		} else {
			result.data[""] = json;
		}
	}

	return result;
}