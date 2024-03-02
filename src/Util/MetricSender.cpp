#include "MetricSender.h"

#include "../../libsrc/cpp_sockets/src/simple_cpp_sockets.h"
#include "../../libsrc/json/nlohmann/json.hpp"

using namespace simple_cpp_sockets;

RLGPC::MetricSender::MetricSender(std::string runName) : runName(runName) {
	udpClient = new UDPClient(PORT);
	RG_LOG("MetricSender initalized for " << (runName.empty() ? "unnamed run" : ("run: \"" + runName + "\"")));
}

void RLGPC::MetricSender::Send(const Report& report) {
	using namespace nlohmann;

	RG_LOG("Sending metrics to socket...");

	DataStreamOut out = {};
	out.Write(COMM_PREFIX);

	lastMsgID++;
	out.Write(lastMsgID);

	// Convert to json
	json j = {};
	j["run_name"] = runName;
	auto& jMetrics = j["metrics"];
	for (auto& pair : report.data)
		jMetrics[pair.first] = pair.second;
	std::stringstream stream;
	stream << j;
	std::string jStr = stream.str();

	out.Write<uint32_t>(jStr.size());
	out.WriteBytes(jStr.data(), jStr.size());

	udpClient->send_message(out.data.data(), out.data.size());

	RG_LOG(" > Done (" << std::fixed << std::setprecision(2) << (out.data.size() / 1000.f) << "KB sent)");
}