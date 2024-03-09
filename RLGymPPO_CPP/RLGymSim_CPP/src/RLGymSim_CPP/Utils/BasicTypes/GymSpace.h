#pragma once
#include "../../Framework.h"

namespace RLGSC {
	// TODO: Temporary 1D array stand-in for an actual implementation of https://gymnasium.farama.org/api/spaces/
	struct GymSpace {
		std::vector<float> data;

		GymSpace() = default;

		GymSpace(size_t size) {
			data = std::vector<float>(size, 0);
		}

		GymSpace(const std::vector<float>& data) :
			data(data) {
		}

		const float* BasePtr() const {
			return data.data();
		}

		float* BasePtr() {
			return data.data();
		}

		float operator[](size_t index) const {
			return data[index];
		}

		float& operator[](size_t index) {
			return data[index];
		}

		GymSpace& operator +=(float val) {
			data.push_back(val);
			return *this;
		}

		GymSpace& operator +=(std::initializer_list<float> floats) {
			data.reserve(floats.size());
			for (float f : floats)
				*this += f;
			return *this;
		}

		GymSpace& operator +=(const Vec& vec) {
			*this += { vec.x, vec.y, vec.z };
			return *this;
		}

		GymSpace& operator +=(const RotMat& mat) {
			for (int i = 0; i < 3; i++)
				*this += mat[i];
			return *this;
		}

		GymSpace& operator +=(const GymSpace& other) {
			data.insert(data.end(), other.data.begin(), other.data.end());
			return *this;
		}
	};
}