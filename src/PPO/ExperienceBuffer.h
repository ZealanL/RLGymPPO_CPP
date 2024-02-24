#pragma once
#include "../Lists.h"

namespace RLGPC {

	struct ExperienceTensors {
		torch::Tensor
			states, actions, logProbs, rewards, nextStates, dones, truncated, values, advantages;

		torch::Tensor* begin() { return &states; }
		torch::Tensor* end() { return &advantages + 1; }
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/experience_buffer.py
	class ExperienceBuffer {
	public:

		torch::Device device;
		int seed;

		ExperienceTensors data;

		int maxSize;

		std::default_random_engine rng;

		ExperienceBuffer(int maxSize, int seed, torch::Device device);

		void SubmitExperience(ExperienceTensors& data);

		struct SampleSet {
			torch::Tensor actions, logProbs, states, values, advantages;
		};
		SampleSet _GetSamples(const int* indices, size_t size) const;

		// Not const because it uses our random engine
		std::vector<SampleSet> GetAllBatchesShuffled(int batchSize);

		void Clear();

		// Combine two tensors into one, removing older data if needed to fit target size
		static torch::Tensor _Concat(torch::Tensor t1, torch::Tensor t2, int size);
	};
}