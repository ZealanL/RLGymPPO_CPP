#include "ExperienceBuffer.h"

#include "../Util/TorchFuncs.h"

using namespace torch;

RLGPC::ExperienceBuffer::ExperienceBuffer(int64_t maxSize, int seed, torch::Device device) :
	maxSize(maxSize), seed(seed), device(device), rng(seed) {
	
}

void RLGPC::ExperienceBuffer::SubmitExperience(ExperienceTensors& _data) {
	RG_NOGRAD;

	bool empty = curSize == 0;

#ifdef RG_PARANOID_MODE
	// Keep copy of target concatination result to keep track of
	auto rewardsTarget = _Concat(
		curSize > 0 ? data.rewards.slice(0, 0, curSize) : data.rewards,
		_data.rewards,
		maxSize
	);
#endif

	for (auto itr1 = data.begin(), itr2 = _data.begin(); itr1 != data.end(); itr1++, itr2++) {
		Tensor& ourTen = *itr1;
		Tensor& addTen = *itr2;

		int64_t addAmount = addTen.size(0);

		if (addAmount > maxSize) {
			addTen = addTen.slice(0, addAmount - maxSize);
			addAmount = maxSize;
		}
		
		int64_t overflow = RS_MAX((curSize + addAmount) - maxSize, 0);
		int64_t startIdx = curSize - overflow;
		int64_t endIdx = curSize + addAmount - overflow;

		if (empty) {
			// Initalize tensor
			
			// Make zero tensor of target size
			auto sizes = addTen.sizes();
			auto newSizes = std::vector<int64_t>(sizes.begin(), sizes.end());
			newSizes[0] = maxSize;
			ourTen = torch::zeros(newSizes);

			// Make ourTen NAN, such that it is obvious if uninitialized data is being used
			ourTen.add_(NAN);

			RG_PARA_ASSERT(ourTen.size(0) == maxSize);
		} else {
			// We already have data
			if (overflow > 0) {
				auto fromData = ourTen.slice(0, overflow, curSize).clone();
				auto toView = ourTen.slice(0, 0, curSize - overflow);
				toView.copy_(fromData);

				RG_PARA_ASSERT(ourTen[curSize - overflow - 1].equal(ourTen[curSize - 1]));
			}
		}

		auto ourTenInsertView = ourTen.slice(0, startIdx, endIdx);
		ourTenInsertView.copy_(addTen);
		RG_PARA_ASSERT(ourTen[endIdx - 1].equal(addTen[addTen.size(0) - 1]));
	}

	curSize = RS_MIN(curSize + _data.begin()->size(0), maxSize);

#ifdef RG_PARANOID_MODE
	// Make sure tensors are all the right size
	for (Tensor& t : data)
		RG_PARA_ASSERT(t.size(0) == maxSize);

	// Make sure our calculation of rewards matches the target
	RG_PARA_ASSERT(data.rewards.slice(0, 0, curSize).equal(rewardsTarget));

	// Make sure that the debug counters go up
	// Games are merged together, meaning the number can reset back down, but never twice in a row
	auto debugCounters = TENSOR_TO_ILIST(data.debugCounters.slice(0, 0, curSize).cpu());
	for (int i = 2; i < debugCounters.size(); i++) {
		if (debugCounters[i] <= debugCounters[i - 1])
			if (debugCounters[i - 1] <= debugCounters[i - 2])
				RG_ERR_CLOSE("Debug counter failed at index " << i);
	}
#endif
}

RLGPC::ExperienceBuffer::SampleSet RLGPC::ExperienceBuffer::_GetSamples(const int64_t* indices, size_t size) const {
	
	// TODO: Slow, use blob
	Tensor tIndices = torch::tensor(IList(indices, indices + size));

	// TODO: Reptitive
	SampleSet result;
	result.actions = torch::index_select(data.actions, 0, tIndices);
	result.logProbs = torch::index_select(data.logProbs, 0, tIndices);
	result.states = torch::index_select(data.states, 0, tIndices);
	result.values = torch::index_select(data.values, 0, tIndices);
	result.advantages = torch::index_select(data.advantages, 0, tIndices);
	return result;
}

std::vector<RLGPC::ExperienceBuffer::SampleSet> RLGPC::ExperienceBuffer::GetAllBatchesShuffled(int64_t batchSize) {

	// Make list of shuffled sample indices
	int64_t* indices = new int64_t[curSize];
	std::iota(indices, indices + curSize, 0); // Fill ascending indices
	std::shuffle(indices, indices + curSize, rng);

	// Get a sample set from each of the batches
	std::vector<SampleSet> result;
	for (int64_t startIdx = 0; startIdx + batchSize <= curSize; startIdx += batchSize) {
		result.push_back(_GetSamples(indices + startIdx, batchSize));
	}

	delete[] indices;
	return result;
}

void RLGPC::ExperienceBuffer::Clear() {
	*this = ExperienceBuffer(maxSize, seed, device);
}

Tensor RLGPC::ExperienceBuffer::_Concat(torch::Tensor t1, torch::Tensor t2, int64_t size) {
	// TODO: torch::cat() is very expensive

	Tensor t;

	int 
		len1 = t1.size(0), 
		len2 = t2.size(0);
	if (len2 > size) {
		// Can't fit all of t2, use the end of it
		t = t2.slice(0, len2 - size);
	} else if (len2 == size) {
		// Can only fit t2
		t = t2;
	} else if (len1 + len2 > size) {
		// Both won't fit, cut off the start of t1 to fit t2
		t = torch::cat({ t1.slice(0, len2 - size), t2 }, 0);
	} else { 
		// Both fit :D
		t = TorchFuncs::ConcatSafe(t1, t2);
	}

	return t;
}