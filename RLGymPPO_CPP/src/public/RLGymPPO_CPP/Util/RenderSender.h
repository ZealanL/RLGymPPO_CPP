#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>

namespace RLGPC {
	struct RenderSender {
		pybind11::module pyMod;

		RenderSender();

		RG_NO_COPY(RenderSender);

		//void Send(const Report& report);

		~RenderSender();
	};
}