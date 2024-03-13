#pragma once
#include "PlayerData.h"
#include "../CommonValues.h"

namespace RLGSC {
	struct ScoreLine {
		int teamGoals[2] = { 0,0 };

		int operator[](size_t index) const {
			return teamGoals[index];
		}

		int& operator[](size_t index) {
			return teamGoals[index];
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/gamestates/game_state.py
	struct GameState {
		ScoreLine scoreLine;
		int lastTouchCarID = -1;
		std::vector<PlayerData> players;

		BallState ballState;
		PhysObj ball, ballInv;

		bool boostPads[CommonValues::BOOST_LOCATIONS_AMOUNT];
		bool boostPadsInv[CommonValues::BOOST_LOCATIONS_AMOUNT];

		// Last arena we updated with
		// Can be used to determine current arena from within reward function, for example
		// NOTE: Could be null
		Arena* lastArena = NULL;

		// Last tick count when updated
		uint64_t lastTickCount = 0;

		GameState() = default;
		explicit GameState(Arena* arena) {
			UpdateFromArena(arena);
		}

		const PhysObj& GetBallPhys(bool inverted) const {
			return inverted ? ballInv : ball;
		}

		const auto& GetBoostPads(bool inverted) const {
			return inverted ? boostPadsInv : boostPads;
		}

		void UpdateFromArena(Arena* arena);
	};
}