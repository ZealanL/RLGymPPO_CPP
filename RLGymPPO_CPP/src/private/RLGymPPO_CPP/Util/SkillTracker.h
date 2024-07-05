#pragma once
#include "../FrameworkTorch.h"
#include "../../../public/RLGymPPO_CPP/Util/SkillTrackerConfig.h"
#include "../../../public/RLGymPPO_CPP/Util/RenderSender.h"
#include "../PPO/DiscretePolicy.h"

#include "../../libsrc/json/nlohmann/json.hpp"

#include <public/RLGymPPO_CPP/Threading/GameInst.h>

namespace RLGPC {
	struct SkillTracker {
		RenderSender* renderSender = NULL;

		struct Game {
			GameInst* gameInst;
			bool teamSwap = false; // To prevent potential bias towards 1 team, the team assignment for old vs current policy is randomized every env reset
			int oldPolicyIndex = 0;

			Game(GameInst* gameInst, int numPolicies) : gameInst(gameInst) {
				Reset(numPolicies);
			}

			void Reset(int numPolicies) {
				teamSwap = RocketSim::Math::RandFloat() > 0.5f;
				oldPolicyIndex = RocketSim::Math::RandInt(0, numPolicies);
			}
		};

		std::vector<Game> games;

		SkillTrackerConfig config;

		struct RatingSet {
			std::map<std::string, float> data;
		};

		RatingSet LoadRatingSet(const nlohmann::json& json, bool warn = true);

		std::vector<DiscretePolicy*> oldPolicies;
		std::vector<RatingSet> oldRatings;
		int64_t timestepsSinceVersionMade = 0;

		uint64_t runCounter = 0;

		std::unordered_set<std::string> modeNames = {};

		RatingSet curRating;

		SkillTracker(const SkillTrackerConfig& config, RenderSender* renderSender = NULL);

		RG_NO_COPY(SkillTracker);

		void RunGames(DiscretePolicy* curPolicy, int64_t timestepsDelta);

		void UpdateRatings(RatingSet& winner, RatingSet& loser, bool updateWinner, bool updateLoser, std::string mode);

		void AppendOldPolicy(DiscretePolicy* policy, RatingSet rating) {
			oldPolicies.push_back(policy);
			oldRatings.push_back(rating);
		}

		~SkillTracker();
	};
}