#pragma once
#include "../../Framework.h"
#include "PhysObj.h"

namespace RLGSC {
	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/gamestates/player_data.py
	struct PlayerData {
		uint32_t carId;
		Team team;

		PhysObj phys, physInv;
		CarState carState;

		int
			matchGoals = -1,
			matchSaves = -1,
			matchShots = -1,
			matchDemos = -1,
			boostPickups = -1;
		bool hasFlip;
		float boostFraction; // From 0 to 1

		bool ballTouchedStep; // True if the player touched the ball during any of tick of the step
		bool ballTouchedTick; // True if the player is touching the ball on the final tick of the step

		void UpdateFromCar(Car* car, uint64_t tickCount, int tickSkip);

		const PhysObj& GetPhys(bool inverted) const {
			return inverted ? physInv : phys;
		}
	};
}