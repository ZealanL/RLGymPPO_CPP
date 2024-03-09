#pragma once
#include "ActionParser.h"

namespace RLGSC {

	// Based on https://github.com/Rolv-Arild/Necto/blob/master/training/parser.py
	class DiscreteAction : public ActionParser {
	public:

		std::vector<Action> actions;

		DiscreteAction();

		virtual ActionSet ParseActions(const Input& actionsData, const GameState& gameState) {
			ActionSet result;
			for (int idx : actionsData)
				result.push_back(actions[idx]);
			return result;
		}

		virtual int GetActionAmount() {
			return actions.size();
		}
	};
}