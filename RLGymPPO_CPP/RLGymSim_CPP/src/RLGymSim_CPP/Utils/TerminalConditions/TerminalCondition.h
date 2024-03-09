#pragma once
#include "../Gamestates/GameState.h"

namespace RLGSC {
	class TerminalCondition {
	public:
		virtual void Reset(const GameState& initialState) {};
		virtual bool IsTerminal(const GameState& currentState) = 0;
	};
}