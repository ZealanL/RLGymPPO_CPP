#include "PlayerData.h"

namespace RLGSC {
	void PlayerData::UpdateFromCar(Car* car, int tickCount) {
		carId = car->id;
		teamNum = car->team;
		carState = car->GetState();
		phys = PhysObj(carState);
		physInv = PhysObj(phys.Invert());

		ballTouched = carState.ballHitInfo.tickCountWhenHit == tickCount;
		hasFlip =
			!carState.isOnGround &&
			!carState.hasDoubleJumped && !carState.hasFlipped
			&& carState.airTimeSinceJump < RLConst::DOUBLEJUMP_MAX_DELAY;

		boostFraction = carState.boost / 100;
	}
}