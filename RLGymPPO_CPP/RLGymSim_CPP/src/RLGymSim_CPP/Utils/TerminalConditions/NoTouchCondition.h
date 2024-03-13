#pragma once
#include "TerminalCondition.h"

namespace RLGSC {
	class NoTouchCondition : public TerminalCondition {
	public:

		int stepsSinceTouch = 0;
		int maxSteps;

		NoTouchCondition(int maxSteps) : maxSteps(maxSteps) {
		}

		virtual void Reset(const GameState& initialState) {
			stepsSinceTouch = 0;
		};

		virtual bool IsTerminal(const GameState& currentState) {
			for (auto& player : currentState.players) {
				if (player.ballTouchedStep) {
					stepsSinceTouch = 0;
					return false;
				}
			}

			stepsSinceTouch++;
			return stepsSinceTouch >= maxSteps;
		}
	};
}