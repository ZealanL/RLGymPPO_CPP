#pragma once
#include "../../Framework.h"
namespace RLGSC {
	struct Quat {
		float w, x, y, z;

		Quat() {
			w = x = y = z = 1;
		}

		Quat(float w, float x, float y, float z) :
			w(w), x(x), y(y), z(z) {
		}

		Quat(const btQuaternion& bulletQuat) :
			w(bulletQuat.w()), x(bulletQuat.x()), y(bulletQuat.y()), z(bulletQuat.z()) {
		}

		static Quat FromRotMat(const RotMat& rotMat);
		RotMat ToRotMat() const;

		explicit operator btQuaternion() const {
			return btQuaternion(x, y, z, w);
		}
	};
}