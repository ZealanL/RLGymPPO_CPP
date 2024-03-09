#include "Match.h"

namespace RLGSC {
	void Match::EpisodeReset(const GameState& initialState) {
		prevActions = ActionSet(initialState.players.size());
		for (auto cond : terminalConditions)
			cond->Reset(initialState);
		rewardFn->Reset(initialState);
		obsBuilder->Reset(initialState);
	}

	FList2 Match::BuildObservations(const GameState& state) {
		auto result = FList2(state.players.size());

		obsBuilder->PreStep(state);

		for (int i = 0; i < state.players.size(); i++) {
			result[i] =
				obsBuilder->BuildOBS(state.players[i], state, prevActions[i]);
		}

		return result;
	}

	std::vector<float> Match::GetRewards(const GameState& state, bool done) {
		auto result = std::vector<float>(state.players.size());

		rewardFn->PreStep(state);

		for (int i = 0; i < state.players.size(); i++) {
			result[i] =
				done ? 
				rewardFn->GetFinalReward(state.players[i], state, prevActions[i]) : 
				rewardFn->GetReward(state.players[i], state, prevActions[i]);
		}

		return result;
	}

	bool Match::IsDone(const GameState& state) {
		for (auto& cond : terminalConditions)
			if (cond->IsTerminal(state))
				return true;

		return false;
	}

	ScoreLine Match::GetScoreLine(const GameState& state) {
		return state.scoreLine;
	}

	ActionSet Match::ParseActions(const ActionParser::Input& actionsData, const GameState& gameState) {
		return actionParser->ParseActions(actionsData, gameState);
	}

	GameState Match::ResetState(Arena* arena) {
		return stateSetter->ResetState(arena);
	}
}