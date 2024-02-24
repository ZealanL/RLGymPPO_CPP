#include "ExperienceBuffer.h"

#include "../Util/TorchFuncs.h"

using namespace torch;

RLGPC::ExperienceBuffer::ExperienceBuffer(int maxSize, int seed, torch::Device device) : 
	maxSize(maxSize), seed(seed), device(device), rng(seed) {
	
}

void RLGPC::ExperienceBuffer::SubmitExperience(ExperienceTensors& _data) {
	for (auto itr1 = data.begin(), itr2 = _data.begin(); itr1 != data.end(); itr1++, itr2++)
		*itr1 = _Concat(*itr1, *itr2, maxSize);
}

RLGPC::ExperienceBuffer::SampleSet RLGPC::ExperienceBuffer::_GetSamples(const int* indices, size_t size) const {
	
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

std::vector<RLGPC::ExperienceBuffer::SampleSet> RLGPC::ExperienceBuffer::GetAllBatchesShuffled(int batchSize) {
	int totalSamples = data.rewards.size(0);

	// Make list of shuffled sample indices
	int* indices = new int[totalSamples];
	std::iota(indices, indices + totalSamples, 0); // Fill ascending indices
	std::shuffle(indices, indices + totalSamples, rng);

	// Get a sample set from each of the batches
	std::vector<SampleSet> result;
	for (int startIdx = 0; startIdx + batchSize <= totalSamples; startIdx += batchSize) {
		result.push_back(_GetSamples(indices + startIdx, batchSize));
	}

	delete[] indices;
	return result;
}

void RLGPC::ExperienceBuffer::Clear() {
	*this = ExperienceBuffer(maxSize, seed, device);
}

Tensor RLGPC::ExperienceBuffer::_Concat(torch::Tensor t1, torch::Tensor t2, int size) {
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