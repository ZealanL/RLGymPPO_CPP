#pragma once
#include "../Utils/Gamestates/GameState.h"
#include "../Utils/BasicTypes/Action.h"
#include "../Utils/TerminalConditions/TerminalCondition.h"
#include "../Utils/RewardFunctions/RewardFunction.h"
#include "../Utils/OBSBuilders/OBSBuilder.h"
#include "../Utils/ActionParsers/ActionParser.h"
#include "../Utils/StateSetters/StateSetter.h"

namespace RLGSC {

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/envs/match.py
	class Match {
	public:
		RewardFunction* rewardFn;
		std::vector<TerminalCondition*> terminalConditions;
		OBSBuilder* obsBuilder;
		ActionParser* actionParser;
		StateSetter* stateSetter;

		int teamSize;
		bool spawnOpponents;
		int playerAmount;

		ActionSet prevActions;

		Match(
			RewardFunction* rewardFn,
			std::vector<TerminalCondition*> terminalConditions,
			OBSBuilder* obsBuilder,
			ActionParser* actionParser,
			StateSetter* stateSetter,
			int teamSize = 1,
			bool spawnOpponents = true
		) : 
			rewardFn(rewardFn),
			terminalConditions(terminalConditions),
			obsBuilder(obsBuilder),
			actionParser(actionParser),
			stateSetter(stateSetter),
			teamSize(teamSize),
			spawnOpponents(spawnOpponents),
			playerAmount(teamSize * (spawnOpponents ? 2 : 1))
		{
			prevActions.resize(playerAmount);
		}

		void EpisodeReset(const GameState& initialState);
		FList2 BuildObservations(const GameState& state);
		FList GetRewards(const GameState& state, bool done);
		bool IsDone(const GameState& state);
		ScoreLine GetScoreLine(const GameState& state);
		ActionSet ParseActions(const ActionParser::Input& actionsData, const GameState& gameState);
		GameState ResetState(Arena* arena);
	};
}