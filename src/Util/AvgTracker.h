#pragma once
#include "../Framework.h"

namespace RLGPC {
	struct AvgTracker {
		float total;
		uint64_t count;

		AvgTracker() {
			Reset();
		}

		// Returns 0 if no count
		float Get() const {
			if (count > 0) {
				return total / count;
			} else {
				return NAN;
			}
		}

		void Add(float val) {
			if (!isnan(val)) {
				total += val;
				count++;
			}
		}

		AvgTracker& operator+=(float val) {
			Add(val);
			return *this;
		}

		void Add(float totalVal, uint64_t count) {
			if (!isnan(totalVal)) {
				total += totalVal;
				this->count += count;
			}
		}

		AvgTracker& operator+=(const AvgTracker& other) {
			Add(other.total, other.count);
			return *this;
		}

		void Reset() {
			total = 0;
			count = 0;
		}
	};
}