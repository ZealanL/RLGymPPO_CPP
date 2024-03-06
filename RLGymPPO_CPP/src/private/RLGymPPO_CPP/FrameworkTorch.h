#pragma once
#include <RLGymPPO_CPP/Framework.h>
#include <RLGymPPO_CPP/Lists.h>

// Include torch
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/utils.h>

#define RG_NOGRAD torch::NoGradGuard _noGradGuard

#define RG_AUTOCAST_ON() at::autocast::set_enabled(true)

#define RG_AUTOCAST_OFF() { \
at::autocast::clear_cache(); \
at::autocast::set_enabled(false); \
}

namespace RLGPC {
	// Method from: https://stackoverflow.com/questions/63466847/how-is-it-possible-to-convert-a-stdvectorstdvectordouble-to-a-torchten
	inline torch::Tensor FLIST2_TO_TENSOR(const FList2& list) {
		int innerSize = list[0].size();
		auto options = torch::TensorOptions().dtype(at::kFloat);
		auto tensor = torch::zeros({ (int)list.size(), innerSize }, options);
		for (int i = 0; i < list.size(); i++)
			// Torch needs the input array to be non-const for whatever reason
			tensor.slice(0, i, i + 1) = torch::from_blob((float*)list[i].data(), { innerSize }, options);

		return tensor;
	}

	inline FList TENSOR_TO_FLIST(torch::Tensor tensor) {
		assert(tensor.dim() == 1);
		tensor = tensor.cpu().detach().to(torch::kFloat32);
		float* data = tensor.data_ptr<float>();
		return FList(data, data + tensor.size(0));
	}

	inline IList TENSOR_TO_ILIST(torch::Tensor tensor) {
		assert(tensor.dim() == 1);
		tensor = tensor.cpu().detach().to(torch::kInt32);
		int* data = tensor.data_ptr<int>();
		return IList(data, data + tensor.size(0));
	}
}