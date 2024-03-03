#pragma once
#include "Report.h"

namespace RLGPC {
	namespace MetricSender {
		constexpr static int
			PORT_CPP = 3942,
			PORT_PY = 3943;
		constexpr static uint32_t
			CONNECT_CODE = 0x1AB80D60,
			COMM_PREFIX  = 0x1AB80D61,
			ACK_PREFIX   = 0x1AB80D62;

		void Init(std::string projectName = {}, std::string groupName = {}, std::string runName = {});
		bool IsConnected();

		void Send(const Report& report);
	};
}