#pragma once
#include "StateSetter.h"

namespace RLGSC {
	class RandomState : public StateSetter {
	public:
		bool
			randBallSpeed, randCarSpeed, carsOnGround;

		RandomState(bool randBallSpeed, bool randCarSpeed, bool carsOnGround) :
			randBallSpeed(randBallSpeed), randCarSpeed(randCarSpeed), carsOnGround(carsOnGround) {
		}

		virtual GameState ResetState(Arena* arena);
	};
}