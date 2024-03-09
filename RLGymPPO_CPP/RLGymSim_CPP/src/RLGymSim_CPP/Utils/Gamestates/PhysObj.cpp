#include "PhysObj.h"

template<typename T>
void _InitPhysObj(RLGSC::PhysObj* obj, const T& rsState) {
	obj->pos = rsState.pos;
	obj->rotMat = rsState.rotMat;
	obj->vel = rsState.vel;
	obj->angVel = rsState.angVel;
}

RLGSC::PhysObj::PhysObj(const BallState& ballState) {
	_InitPhysObj(this, ballState);
}

RLGSC::PhysObj::PhysObj(const CarState& carState) {
	_InitPhysObj(this, carState);
}

RLGSC::PhysObj RLGSC::PhysObj::Invert() {
	PhysObj result = *this;

	constexpr Vec invVec = Vec(-1, -1, 1);

	result.pos *= invVec;
	for (int i = 0; i < 3; i++)
		result.rotMat[i] *= invVec;
	result.vel *= invVec;
	result.angVel *= invVec;

	return result;
}