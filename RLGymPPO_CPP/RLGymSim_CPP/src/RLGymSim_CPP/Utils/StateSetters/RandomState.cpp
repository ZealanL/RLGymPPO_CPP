#include "RandomState.h"
#include "../../Math.h"

RLGSC::GameState RLGSC::RandomState::ResetState(Arena* arena) {
	
	constexpr float
		X_MAX = 3500,
		Y_MAX = 4000,
		Z_MAX = 1820,
		PITCH_MAX = M_PI / 2,
		YAW_MAX = M_PI,
		ROLL_MAX = M_PI;

	{ // Randomize ball
		BallState bs = {};
		bs.pos = Math::RandVec(Vec(-X_MAX, -Y_MAX, CommonValues::BALL_RADIUS), Vec(X_MAX, Y_MAX, Z_MAX));
		if (randBallSpeed) {
			bs.vel = Math::RandVec(Vec(-3000, -3000, -3000), Vec(3000, 3000, 3000));
			bs.angVel = Math::RandVec(Vec(-4, -4, -4), Vec(4, 4, 4));
		}
		arena->ball->SetState(bs);
	}

	for (Car* car : arena->_cars) { // Randomize cars
		CarState cs = {};
		bool onGround = carsOnGround ? true : (::Math::RandFloat() > 0.5);
		cs.pos = Math::RandVec(Vec(-X_MAX, -Y_MAX, 40), Vec(X_MAX, Y_MAX, Z_MAX));

		Angle angle = Angle(::Math::RandFloat(-YAW_MAX, YAW_MAX), ::Math::RandFloat(-PITCH_MAX, PITCH_MAX), ::Math::RandFloat(-ROLL_MAX, ROLL_MAX));

		if (onGround) {
			cs.pos.z = 17;
			angle.pitch = angle.roll = 0;
		}

		cs.rotMat = angle.ToRotMat();

		if (randCarSpeed) {
			// Might go outside of max vel but I do not care
			cs.vel = Math::RandVec(Vec(-2300, -2300, -2300), Vec(2300, 2300, 2300));
			cs.angVel = Math::RandVec(Vec(-4, -4, -4), Vec(4, 4, 4));
		}
		car->SetState(cs);
	}

	return GameState(arena);
}