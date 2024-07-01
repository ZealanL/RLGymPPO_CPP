#include "SkillTracker.h"
#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Math.h>

using namespace RLGSC;

class DummyReward : public RLGSC::RewardFunction {
	virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
		return 0;
	}
};

static thread_local DummyReward* g_DummyReward = new DummyReward();

RLGPC::SkillTracker::SkillTracker(const SkillTrackerConfig& config, RenderSender* renderSender) 
	: config(config), renderSender(renderSender) {

	RG_ASSERT(config.numEnvs > 0);
	RG_ASSERT(config.timestepsPerVersion >= 0);
	RG_ASSERT(config.maxVersions > 0);
	RG_ASSERT(config.simTime > 0);

	curRating = config.initialRating;

	for (int i = 0; i < config.numEnvs; i++) {

		RG_ASSERT(config.envCreateFunc);

		auto envCreateResult = config.envCreateFunc();
		GameInst* gameInst = new GameInst(envCreateResult.gym, envCreateResult.match);
		gameInst->match->rewardFn = g_DummyReward;
		gameInst->Start();
		games.push_back(Game(gameInst, 1));
	}
}

RLGPC::SkillTracker::~SkillTracker() {
	for (auto& policy : oldPolicies)
		delete policy;

	for (auto game : games)
		delete game.gameInst;
}

void RLGPC::SkillTracker::UpdateRatings(float& winner, float& loser) {
	// Simple elo calculation

	float expDelta = (loser - winner) / 400;
	float expected = 1 / (powf(10, expDelta) + 1);

	winner += config.ratingInc * (1 - expected);
	loser += config.ratingInc * (expected - 1);
}

void RLGPC::SkillTracker::RunGames(DiscretePolicy* curPolicy, int64_t timestepsDelta) {
	constexpr const char* ERR_PREFIX = "RLGPC::SkillTracker::RunGames(): ";

	if (oldPolicies.empty() && config.startWithVersion) {
		DiscretePolicy* newOldPolicy = new DiscretePolicy(curPolicy->inputAmount, curPolicy->actionAmount, curPolicy->layerSizes, curPolicy->device);
		curPolicy->CopyTo(*newOldPolicy);
		oldPolicies.push_back(newOldPolicy);
		oldRatings.push_back(curRating);
	}

	if (oldPolicies.size() > 0) {
		float prevRating = curRating;

		float timePerGame = config.simTime / games.size();

		for (int gameIdx = 0; gameIdx < games.size(); gameIdx++) {
			auto& game = games[gameIdx];
			auto& gameInst = game.gameInst;
			gameInst->stepCallback = config.stepCallback;

			DiscretePolicy* oldPolicy = oldPolicies[game.oldPolicyIndex];

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
					if (scoringPolicy == curPolicy) {
						// Current policy scored
						UpdateRatings(curRating, oldRatings[game.oldPolicyIndex]);
					} else {
						// Old policy scored
						UpdateRatings(oldRatings[game.oldPolicyIndex], curRating);
					}
				}

				if (stepResult.done)
					game.Reset(oldPolicies.size());

				if (renderSender) {
					renderSender->Send(stepResult.state, gameInst->match->prevActions);
					float sleepTime = gameInst->gym->tickSkip / 120.f;
					std::this_thread::sleep_for(std::chrono::microseconds(int64_t(sleepTime * 1000 * 1000)));
				}
			}
		}

		lastRatingDelta = curRating - prevRating;
		RG_LOG(
			" > New rating: " << std::setprecision(6) << curRating << 
			" (" << (lastRatingDelta >= 0 ? "+" : "") << std::setprecision(4) << lastRatingDelta << ")"
		);
	} else {
		RG_LOG(" > No old policies yet, skipping");
		lastRatingDelta = 0;
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