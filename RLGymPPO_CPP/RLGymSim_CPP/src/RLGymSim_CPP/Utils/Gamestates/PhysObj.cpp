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

RLGSC::PhysObj RLGSC::PhysObj::Invert() const {
	PhysObj result = *this;

	constexpr Vec invVec = Vec(-1, -1, 1);

	result.pos *= invVec;
	for (int i = 0; i < 3; i++)
		result.rotMat[i] *= invVec;
	result.vel *= invVec;
	result.angVel *= invVec;

	return result;
}

RLGSC::PhysObj RLGSC::PhysObj::MirrorX() const {
	PhysObj result = *this;

	result.pos.x *= -1;

	// Thanks Rolv, JPK, and Kaiyo!
	result.rotMat.forward *= Vec(-1,  1,  1);
	result.rotMat.right   *= Vec( 1, -1, -1);
	result.rotMat.up      *= Vec(-1,  1,  1);

	result.vel.x *= -1;
	result.angVel *= Vec(1, -1, -1);

	return result;
}