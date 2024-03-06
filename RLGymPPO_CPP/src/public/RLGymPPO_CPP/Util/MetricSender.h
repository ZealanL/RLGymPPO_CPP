#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>

namespace RLGPC {
	struct MetricSender {
		std::string curRunID;
		std::string projectName, groupName, runName;
		pybind11::module pyMod;

		MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {}, std::string runID = {});
		
		RG_NO_COPY(MetricSender);

		void Send(const Report& report);

		~MetricSender();
	};
}