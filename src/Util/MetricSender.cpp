#include "MetricSender.h"

#include "../../libsrc/cpp_sockets/src/simple_cpp_sockets.h"
#include "../../libsrc/json/nlohmann/json.hpp"
#include "Timer.h"

using namespace simple_cpp_sockets;
using namespace RLGPC::MetricSender;

std::string projectName, groupName, runName;

simple_cpp_sockets::UDPClient* udpClient;
simple_cpp_sockets::UDPServer* udpServer;
uint32_t lastMsgID = 0;
std::atomic<bool> initialized = false, connected = false, bound = false;

std::thread receiverThread, udpServerThread;

void _ReceiverScriptThread() {
#ifdef WIN32
	constexpr char DIR_SEPARATOR = '\\';
#else
	constexpr char DIR_SEPARATOR = '/';
#endif

	std::string cmd = RS_STR("\"python_scripts" << DIR_SEPARATOR << "metrics_receiver.py\" " << projectName << " " << groupName << " " << runName);
#ifndef WIN32
	cmd = "python " + cmd;
#endif
	system(cmd.c_str());

	RG_ERR_CLOSE("Metric receiver died (see above)");
}

// To receive acknowledgements from server
void _UDPServerThread() {
	
	udpServer = new UDPServer(PORT_CPP);
	int result = udpServer->socket_bind();
	if (result)
		RG_ERR_CLOSE("MetricSender: Failed to bind UDP server, result=" << result);

	bound = true;

	while (!connected) {
		try {
			udpServer->listen(1024,
				[](void* data, size_t size) {
					RG_LOG("Recieved " << size << " bytes...");

					if (size == sizeof(uint32_t)) {
						// TODO: Validate code
						connected = true;
					}
				}
			);
		} catch (std::exception& e) {
			RG_ERR_CLOSE("Socket listen exception: " << e.what());
		}
	}
	RG_LOG("Connected to metrics receiver!");
}

void RLGPC::MetricSender::Init(std::string _projectName, std::string _groupName, std::string _runName) {

	if (!initialized) {
		initialized = true;
	} else {
		RG_ERR_CLOSE("MetricSender::Init() called multiple times");
	}

	projectName = _projectName;
	groupName = _groupName;
	runName = _runName;

	udpServerThread = std::thread(_UDPServerThread);
	udpServerThread.detach();

	while (!bound)
		std::this_thread::yield();

	receiverThread = std::thread(_ReceiverScriptThread);
	receiverThread.detach();

	udpClient = new UDPClient(PORT_PY);
	RG_LOG("MetricSender initalized for " << (runName.empty() ? "unnamed run" : ("run: \"" + runName + "\"")));
}

bool RLGPC::MetricSender::IsConnected() {
	return connected;
}

void RLGPC::MetricSender::Send(const Report& report) {
	using namespace nlohmann;

	{ // Wait for connection
		constexpr float MAX_WAIT_TIME = 5;
		Timer waitTimer = {};
		while (!connected) {
			std::this_thread::yield();

			if (waitTimer.Elapsed() > MAX_WAIT_TIME)
				RG_ERR_CLOSE(
					"Timed out while waiting to connect to metrics receiver.\n" <<
					"Make sure the metrics receiver was started succesfully."
				);
		}
	}

	RG_LOG("Sending metrics to socket...");

	DataStreamOut out = {};
	out.Write(COMM_PREFIX);

	lastMsgID++;
	out.Write(lastMsgID);

	// Convert to json
	json j = {};
	j["project_name"] = projectName;
	j["group_name"] = groupName;
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