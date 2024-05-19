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

	FList Match::GetRewards(const GameState& state, bool done) {
		auto result = FList(state.players.size());

		rewardFn->PreStep(state);
		return rewardFn->GetAllRewards(state, prevActions, done);
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
		ActionSet actions = actionParser->ParseActions(actionsData, gameState);

		for (int i = 0; i < gameState.players.size(); i++)
			if (gameState.players[i].carState.isDemoed)
				actions[i] = {};

		return actions;
	}

	GameState Match::ResetState(Arena* arena) {
		GameState newState = stateSetter->ResetState(arena);

		if (newState.players.size() != playerAmount) {
			RG_ERR_CLOSE(
				"Match::ResetState(): New state has a different amount of players, "
				"expected " << playerAmount << " but got " << newState.players.size() << ".\n"
				"Changing number of players at state reset is currently not supported.\n" <<
				"If you want variable player amounts, set a differing player amount per env."
			);
		}

		for (auto& pad : arena->_boostPads)
			pad->SetState({});

		return newState;
	}
}