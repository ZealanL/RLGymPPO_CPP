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

		// matchAssists: being the passer to a teammate who shot and scored
		// matchBumps: any bump against an opponent, including demos
		int
			matchGoals = 0,
			matchSaves = 0,
			matchShots = 0,
			matchAssists = 0,
			matchBumps = 0, 
			matchDemos = 0,
			boostPickups = 0;
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