#include "CommonRewards.h"
#include "CommonRewards.h"
#include "CommonRewards.h"
#pragma once
#include "CommonRewards.h"

RLGSC::EventReward::EventReward(WeightScales weightScales) {
	for (int i = 0; i < ValSet::VAL_AMOUNT; i++)
		weights[i] = weightScales[i];
}

RLGSC::EventReward::ValSet RLGSC::EventReward::ExtractValues(const PlayerData& player, const GameState& state) {
	ValSet result = {};

	int teamGoals = state.scoreLine[(int)player.team],
		opponentGoals = state.scoreLine[1 - (int)player.team];

	float newVals[ValSet::VAL_AMOUNT] = {
		player.matchGoals, teamGoals, opponentGoals, player.ballTouchedStep, player.matchShots, player.matchSaves, player.matchDemos, player.carState.isDemoed, player.boostFraction
	};

	memcpy(result.vals, newVals, sizeof(newVals));
	return result;
}

void RLGSC::EventReward::Reset(const GameState& state) {
	lastRegisteredValues = {};
	for (auto& player : state.players)
		lastRegisteredValues[player.carId] = ExtractValues(player, state);
}

float RLGSC::EventReward::GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
	auto& oldValues = lastRegisteredValues[player.carId];
	auto newValues = ExtractValues(player, state);

	float reward = 0;
	for (int i = 0; i < ValSet::VAL_AMOUNT; i++)
		reward += RS_MAX(newValues[i] - oldValues[i], 0) * weights[i];

	oldValues = newValues;
	return reward;
}