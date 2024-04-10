#pragma once
#include "StateSetter.h"

namespace RLGSC {
	class KickoffState : public StateSetter {
	public:
		virtual GameState ResetState(Arena* arena) {
			arena->ResetToRandomKickoff();
			return GameState(arena);
		}
	};
}