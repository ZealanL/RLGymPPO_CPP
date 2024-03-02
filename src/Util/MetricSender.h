#pragma once
#include "Report.h"

// Forward declaration
namespace simple_cpp_sockets { class UDPClient; }

namespace RLGPC {
	struct MetricSender {

		constexpr static int PORT = 3942;
		constexpr static uint32_t COMM_PREFIX = 0x1AB80D64;

		std::string runName;
		simple_cpp_sockets::UDPClient* udpClient;
		uint32_t lastMsgID = 0;

		MetricSender(std::string runName = {});

		void Send(const Report& report);

		RG_NO_COPY(MetricSender);

		~MetricSender() {
			delete udpClient;
		}
	};
}