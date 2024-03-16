#pragma once
#include "../Util/TorchFuncs.h"

namespace RLGPC {
	struct TrajectoryTensors {
		torch::Tensor
			states,
			actions,
			logProbs,
			rewards,

#ifdef RG_PARANOID_MODE
			debugCounters,
#endif
			nextStates,
			dones,
			truncateds;

		constexpr static size_t TENSOR_AMOUNT =
#ifdef RG_PARANOID_MODE
			8;
#else
			7;
#endif

		torch::Tensor* begin() { return &states; }
		const torch::Tensor* begin() const { return &states; }
		torch::Tensor* end() { return &states + TENSOR_AMOUNT; }
		const torch::Tensor* end() const { return &states + TENSOR_AMOUNT; }

		torch::Tensor& operator[](size_t index) { return *(begin() + index); }
		const torch::Tensor& operator[](size_t index) const { return *(begin() + index); }
	};

	// A container for the timestep data of a specific agent
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/batched_agents/batched_trajectory.py
	// Unlike rlgym-ppo, this has a capacity allocation system like std::vector
	// This class is designed to append single steps or merge multiple trajectories as fast as possible
	struct GameTrajectory {

		TrajectoryTensors data;
		size_t size = 0, capacity = 0;

#ifdef RG_PARANOID_MODE
		int64_t debugCounter = 0;
#endif

		void Append(GameTrajectory& other);
		void MultiAppend(const std::vector<GameTrajectory>& others); // Much faster than spamming Append()

		void RemoveCapacity();

		// NOTE: Assumes the passed in data is a SINGLE TIMESTEP, at a lower dimensionality than multiple timesteps,
		//	e.g. data.rewards should actually just be a single float tensor
		void AppendSingleStep(TrajectoryTensors step);

		void Clear() {
#ifdef RG_PARANOID_MODE
			int64_t debugCounterBackup = debugCounter;
#endif
			*this = GameTrajectory();

#ifdef RG_PARANOID_MODE
			debugCounter = debugCounterBackup;
#endif
		}

		// Doubles the capacity
		void DoubleReserve();
	};
}