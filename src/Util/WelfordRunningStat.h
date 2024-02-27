#pragma once
#include "../Framework.h"

namespace RLGPC {
	struct WelfordRunningStat {
		FList ones, zeros;
		FList runningMean, runningVariance;

		int count, shape;

		WelfordRunningStat() = default;
		WelfordRunningStat(int shape) {
			this->ones = FList(shape);
			std::fill(ones.begin(), ones.end(), 1);
			this->zeros = FList(shape);

			this->runningMean = FList(shape);
			this->runningVariance = FList(shape);

			this->count = 0;
			this->shape = shape;
		}

		void Increment(const FList2& samples, int num) {
			for (int i = 0; i < num; i++)
				Update(samples[i]);
		}

		// TODO: Inefficient construction of another FList
		void Increment(const FList& samples, int num) {
			for (int i = 0; i < num; i++)
				Update(FList({ samples[i] }));
		}

		void Update(const FList& sample) {
			int currentCount = count;
			count++;

			// TODO: Inefficient, only need 1 loop

			FList delta = FList(shape);
			FList deltaN = FList(shape); 
			for (int i = 0; i < shape; i++) {
				delta[i] = sample[i] - runningMean[i];
				deltaN[i] = delta[i] / count;
			}

			for (int i = 0; i < shape; i++) {
				runningMean[i] += deltaN[i];
				runningMean[i] += delta[i] * deltaN[i] * currentCount;
			}
		}

		void Reset() {
			*this = WelfordRunningStat(shape);
		}

		FList Mean() {
			if (count < 2)
				return zeros;

			return runningMean;
		}

		FList GetSTD() {
			if (count < 2)
				return ones;

			FList var = FList(runningVariance.size());
			for (int i = 0; i < var.size(); i++) {
				float curVar = runningVariance[i] / (count - 1);
				if (curVar == 0)
					curVar = 1;
				var[i] = sqrt(curVar);
			}

			return var;
		}


	};
}