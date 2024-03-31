#pragma once
#include "../Framework.h"

namespace RLGPC {
	struct WelfordRunningStat {
		FList ones, zeros;
		std::vector<double> runningMean, runningVariance;

		int64_t count, shape;

		WelfordRunningStat() = default;
		WelfordRunningStat(int shape) {
			this->ones = FList(shape);
			this->zeros = FList(shape);
			std::fill(ones.begin(), ones.end(), 1);
			std::fill(zeros.begin(), zeros.end(), 1);

			this->runningMean = std::vector<double>(shape);
			this->runningVariance = std::vector<double>(shape);

			this->count = 0;
			this->shape = shape;
		}

		void Increment(const FList2& samples, int num) {
			for (int i = 0; i < num; i++)
				Update(samples[i]);
		}

		// TODO: Inefficient construction of another DList
		void Increment(const FList& samples, int num) {
			for (int i = 0; i < num; i++)
				Update(FList({ samples[i] }));
		}

		void Update(const FList& sample) {
			int64_t currentCount = count;
			count++;

			// TODO: Inefficient, only need 1 loop

			auto delta = std::vector<double>(shape);
			auto deltaN = std::vector<double>(shape);
			for (int i = 0; i < shape; i++) {
				delta[i] = sample[i] - runningMean[i];
				deltaN[i] = delta[i] / count;
			}

			for (int i = 0; i < shape; i++) {
				runningMean[i] += deltaN[i];
				runningVariance[i] += delta[i] * deltaN[i] * currentCount;
			}
		}

		void Reset() {
			*this = WelfordRunningStat(shape);
		}

		FList Mean() const {
			if (count < 2)
				return zeros;

			FList runningMeanF = {};
			for (double d : runningMean)
				runningMeanF.push_back(d);

			return runningMeanF;
		}

		FList GetSTD() const {
			if (count < 2)
				return ones;

			FList var = FList(runningVariance.size());
			for (int i = 0; i < var.size(); i++) {
				double curVar = runningVariance[i] / (count - 1);
				if (curVar == 0)
					curVar = 1;
				var[i] = sqrt(curVar);
			}

			return var;
		}
	};
}