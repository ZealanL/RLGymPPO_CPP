#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>

namespace RLGPC {
	struct MetricSender {
		constexpr static int
			PORT_CPP = 3942,
			PORT_PY = 3943;
		constexpr static uint32_t
			CONNECT_CODE = 0x1AB80D60,
			COMM_PREFIX  = 0x1AB80D61,
			ACK_PREFIX   = 0x1AB80D62;

		std::string projectName, groupName, runName;
		pybind11::module pyMod;

		MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {});
		
		RG_NO_COPY(MetricSender);

		void Send(const Report& report);

		~MetricSender();
	};
}