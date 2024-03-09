#include "Math.h"

bool RLGSC::Math::IsBallScored(Vec pos) {
	return abs(pos.y) > RLConst::SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y + RLConst::BALL_COLLISION_RADIUS_SOCCAR;
}

Vec RLGSC::Math::RandVec(Vec min, Vec max) {
	return Vec(
		::Math::RandFloat(min.x, max.x),
		::Math::RandFloat(min.y, max.y),
		::Math::RandFloat(min.z, max.z)
	);
}
