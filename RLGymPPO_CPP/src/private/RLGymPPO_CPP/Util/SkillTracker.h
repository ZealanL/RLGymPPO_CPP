#pragma once
#include "../FrameworkTorch.h"
#include "../../../public/RLGymPPO_CPP/Util/SkillTrackerConfig.h"
#include "../../../public/RLGymPPO_CPP/Util/RenderSender.h"
#include "../PPO/DiscretePolicy.h"

#include <public/RLGymPPO_CPP/Threading/GameInst.h>

namespace RLGPC {
	struct SkillTracker {
		RenderSender* renderSender = NULL;

		std::vector<GameInst*> games;

		// To prevent potential bias towards 1 team, the team assignment for old vs current policy is randomized every env reset
		std::vector<bool> gameTeamSwaps;

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