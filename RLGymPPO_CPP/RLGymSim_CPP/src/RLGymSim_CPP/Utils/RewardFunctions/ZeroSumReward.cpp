#include "ZeroSumReward.h"

std::vector<float> RLGSC::ZeroSumReward::GetAllRewards(const GameState& state, const ActionSet& prevActions, bool final) {
	std::vector<float> rewards = childFunc->GetAllRewards(state, prevActions, final);

	int teamCounts[2] = {};
	float avgTeamRewards[2] = {};

	for (int i = 0; i < state.players.size(); i++) {
		int teamIdx = (int)state.players[i].team;
		teamCounts[teamIdx]++;
		avgTeamRewards[teamIdx] += rewards[i];
	}

	for (int i = 0; i < 2; i++)
		avgTeamRewards[i] /= RS_MAX(teamCounts[i], 1);

	for (int i = 0; i < state.players.size(); i++) {
		auto& player = state.players[i];
		int teamIdx = (int)player.team;
		int teamCount = teamCounts[teamIdx];

		rewards[i] =
			rewards[i] * (1 - teamSpirit)
			+ (avgTeamRewards[teamIdx] * teamSpirit)
			- (avgTeamRewards[1 - teamIdx] * opponentScale);
	}

	return rewards;
}