#include "Quat.h"

namespace RLGSC {
	Quat Quat::FromRotMat(const RotMat& rotMat) {
		btQuaternion quat;
		((btMatrix3x3)rotMat).getRotation(quat);
		return Quat(quat);
	}

	RotMat Quat::ToRotMat() const {
		btMatrix3x3 rotMat;
		rotMat.setRotation((btQuaternion)*this);
		return rotMat;
	}
}