#pragma once
#include "../../Framework.h"

namespace RLGSC {
	struct Action {
		float
			throttle, steer,
			pitch, yaw, roll,
			boost, jump, handbrake;
		constexpr static size_t ELEM_AMOUNT = 8;

		Action() = default;

		Action(const CarControls& controls) {
			throttle = controls.throttle;
			steer = controls.steer;
			pitch = controls.pitch;
			yaw = controls.yaw;
			roll = controls.roll;
			boost = controls.boost;
			jump = controls.jump;
			handbrake = controls.handbrake;
		}

		Action(
			float throttle, float steer,
			float pitch, float yaw, float roll,
			float boost, float jump, float handbrake
		) : throttle(throttle), steer(steer), 
			pitch(pitch), yaw(yaw), roll(roll),
			boost(boost), jump(jump), handbrake(handbrake) {
		}

		explicit operator CarControls() const {
			CarControls result = {};
			result.throttle = throttle;
			result.steer = steer;
			result.pitch = pitch;
			result.yaw = yaw;
			result.roll = roll;

			result.boost = boost == 1;
			result.jump = jump == 1;
			result.handbrake = handbrake == 1;
			return result;
		}

		float* begin() { return &throttle; }
		float* end() { return &handbrake + 1; }

		const float* begin() const { return &throttle; }
		const float* end() const { return &handbrake + 1; }

		float operator[](size_t index) const {
			assert(index < ELEM_AMOUNT);
			return begin()[index];
		}

		float& operator[](size_t index) {
			assert(index < ELEM_AMOUNT);
			return begin()[index];
		}
	};

	typedef std::vector<Action> ActionSet;
}