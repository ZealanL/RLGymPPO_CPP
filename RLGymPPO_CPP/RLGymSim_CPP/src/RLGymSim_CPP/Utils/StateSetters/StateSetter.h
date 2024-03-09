#pragma once
#include "../Gamestates/GameState.h"

namespace RLGSC {
	class StateSetter {
	public:

		// NOTE: Applies reset state to arena
		virtual GameState ResetState(Arena* arena) = 0;
	};
}