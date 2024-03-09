#pragma once
#include "TerminalCondition.h"
#include "../../Math.h"

namespace RLGSC {
	class GoalScoreCondition : public TerminalCondition {
	public:

		virtual bool IsTerminal(const GameState& currentState) {
			return Math::IsBallScored(currentState.ball.pos);
		}
	};
}