#include "PlayerData.h"

namespace RLGSC {
	void PlayerData::UpdateFromCar(Car* car, uint64_t tickCount, int tickSkip) {
		carId = car->id;
		team = car->team;
		CarState newState = car->GetState();

		if (newState.isDemoed && newState.pos.z == -50000) {
			newState.pos = carState.pos;
			newState.vel = carState.vel;
			newState.angVel = carState.angVel;
		}

		carState = newState;

		phys = PhysObj(carState);
		physInv = PhysObj(phys.Invert());

		if (carState.ballHitInfo.isValid) {
			ballTouchedStep = carState.ballHitInfo.tickCountWhenHit >= (tickCount - tickSkip);
			ballTouchedTick = carState.ballHitInfo.tickCountWhenHit == (tickCount - 1);
		} else {
			ballTouchedStep = ballTouchedTick = false;
		}

		hasJump = !carState.hasJumped;
		hasFlip =
			!carState.hasDoubleJumped && !carState.hasFlipped
			&& carState.airTimeSinceJump < RLConst::DOUBLEJUMP_MAX_DELAY;

		boostFraction = carState.boost / 100;
	}
}