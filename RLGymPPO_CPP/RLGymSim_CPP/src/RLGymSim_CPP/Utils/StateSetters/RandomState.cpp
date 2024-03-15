#include "RandomState.h"
#include "../../Math.h"

Vec RandNormVec() {
	return RLGSC::Math::RandVec(Vec(-1, -1, -1), Vec(1, 1, 1)).Normalized();
}

RLGSC::GameState RLGSC::RandomState::ResetState(Arena* arena) {
	
	constexpr float
		X_MAX = 3500,
		Y_MAX = 4000,
		Z_MAX = 1820,
		CAR_Z_MIN = 150,
		PITCH_MAX = M_PI / 2,
		YAW_MAX = M_PI,
		ROLL_MAX = M_PI,
		ANGVEL_MAX = 5.5f;

	{ // Randomize ball
		BallState bs = {};
		bs.pos = Math::RandVec(Vec(-X_MAX, -Y_MAX, CommonValues::BALL_RADIUS), Vec(X_MAX, Y_MAX, Z_MAX));
		if (randBallSpeed) {
			bs.vel = RandNormVec() * ::Math::RandFloat(0, 4000);
			bs.angVel = Math::RandVec(Vec(-4, -4, -4), Vec(4, 4, 4));
		}
		arena->ball->SetState(bs);
	}

	for (Car* car : arena->_cars) { // Randomize cars
		CarState cs = {};
		cs.pos = Math::RandVec(Vec(-X_MAX, -Y_MAX, CAR_Z_MIN), Vec(X_MAX, Y_MAX, Z_MAX));

		if (randCarSpeed) {
			// Might go outside of max vel but I do not care
			Vec randVelDir = Math::RandVec(Vec(-1, -1, -1), Vec(1, 1, 1)).Normalized();
			cs.vel = RandNormVec() * ::Math::RandFloat(0, RLConst::CAR_MAX_SPEED);
			cs.angVel = RandNormVec() * ANGVEL_MAX;
		}

		Angle angle = Angle(::Math::RandFloat(-YAW_MAX, YAW_MAX), ::Math::RandFloat(-PITCH_MAX, PITCH_MAX), ::Math::RandFloat(-ROLL_MAX, ROLL_MAX));

		bool onGround = carsOnGround ? true : (::Math::RandFloat() > 0.5);
		if (onGround) {
			cs.pos.z = 17;
			angle.pitch = angle.roll = 0;
			cs.vel.z = 0;
			cs.angVel = {};
		}

		cs.rotMat = angle.ToRotMat();
		car->SetState(cs);
	}

	return GameState(arena);
}