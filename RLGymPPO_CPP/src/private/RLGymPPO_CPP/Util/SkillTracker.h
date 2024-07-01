#pragma once
#include "../FrameworkTorch.h"
#include "../../../public/RLGymPPO_CPP/Util/SkillTrackerConfig.h"
#include "../../../public/RLGymPPO_CPP/Util/RenderSender.h"
#include "../PPO/DiscretePolicy.h"

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

		std::vector<DiscretePolicy*> oldPolicies;
		std::vector<float> oldRatings;
		int64_t timestepsSinceVersionMade = 0;

		float curRating;
		float lastRatingDelta = 0;

		SkillTracker(const SkillTrackerConfig& config, RenderSender* renderSender = NULL);

		RG_NO_COPY(SkillTracker);

		void RunGames(DiscretePolicy* curPolicy, int64_t timestepsDelta);

		void UpdateRatings(float& winner, float& loser);

		~SkillTracker();
	};
}