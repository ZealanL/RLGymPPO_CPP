#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/reward_function.py
namespace RLGSC {
	class RewardFunction {
	public:
		virtual void Reset(const GameState& initialState) {}

		virtual void PreStep(const GameState& state) {}

		virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
			throw std::runtime_error("GetReward() is unimplemented");
			return 0;
		}

		virtual float GetFinalReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
			return GetReward(player, state, prevAction);
		}

		// Get all rewards for all players
		virtual std::vector<float> GetAllRewards(const GameState& state, const ActionSet& prevActions, bool final) {

			std::vector<float> rewards = std::vector<float>(state.players.size());
			for (int i = 0; i < state.players.size(); i++) {
				if (final) {
					rewards[i] = GetFinalReward(state.players[i], state, prevActions[i]);
				} else {
					rewards[i] = GetReward(state.players[i], state, prevActions[i]);
				}
			}

			return rewards;
		}

		virtual ~RewardFunction() {};
	};
}