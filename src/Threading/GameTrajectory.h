#pragma once
#include "../Util/TorchFuncs.h"

namespace RLGPC {
	struct TrajectoryTensors {
		torch::Tensor
			states,
			actions,
			logProbs,
			rewards,
			nextStates,
			dones,
			truncateds;
		torch::Tensor* begin() {
			return &states;
		}

		torch::Tensor* end() {
			return &truncateds + 1;
		}
	};

	// A container for the timestep data of a specific agent
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/batched_agents/batched_trajectory.py
	// Unlike rlgym-ppo, this has a capacity allocation system like std::vector
	// This makes adding single steps to a trajectory substantially faster
	struct GameTrajectory {

		TrajectoryTensors data;
		size_t size = 0, capacity = 0;

		void Append(GameTrajectory& other);

		void RemoveCapacity();

		// NOTE: Assumes the passed in data is a SINGLE TIMESTEP, at a lower dimensionality than multiple timesteps,
		//	e.g. data.rewards should actually just be a single float tensor
		void AppendSingleStep(TrajectoryTensors step);

		void Clear() {
			*this = GameTrajectory();
		}

		// Doubles the capacity
		void DoubleReserve();
	};
}