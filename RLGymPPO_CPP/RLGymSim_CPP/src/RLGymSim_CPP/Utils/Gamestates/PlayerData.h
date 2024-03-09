#pragma once
#include "../../Framework.h"
#include "PhysObj.h"

namespace RLGSC {
	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/gamestates/player_data.py
	struct PlayerData {
		uint32_t carId;
		Team teamNum;

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

		bool ballTouched;

		void UpdateFromCar(Car* car, int tickCount);

		const PhysObj& GetPhys(bool inverted) const {
			return inverted ? physInv : phys;
		}
	};
}