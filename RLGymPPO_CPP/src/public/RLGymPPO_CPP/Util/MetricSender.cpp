#include "MetricSender.h"

#include "Timer.h"
#include <csignal>

namespace py = pybind11;
using namespace RLGPC;

RLGPC::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender...");

	try {
		pyMod = py::module::import("python_scripts.metric_receiver");
		this->initMethod = pyMod.attr("init");
		this->sendMetricsMethod = pyMod.attr("add_metrics");
		this->onKillMethod = pyMod.attr("end");
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to import metrics receiver, exception: " << e.what());
	}

	try {
		auto returedRunID = this->initMethod(PY_EXEC_PATH, projectName, groupName, runName, runID);
		curRunID = returedRunID.cast<std::string>();
		RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing") << " run with ID : \"" << curRunID << "\"...");

	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to initialize in Python, exception: " << e.what());
	}

	RG_LOG(" > MetricSender initalized.");

	std::signal(SIGINT, MetricSender::OnKillSignal);
	std::signal(SIGTERM, MetricSender::OnKillSignal);
	std::signal(SIGBREAK, MetricSender::OnKillSignal);
}

void RLGPC::MetricSender::Send(const Report& report) {
	py::dict reportDict = {};

	for (auto& pair : report.data)
		reportDict[pair.first.c_str()] = pair.second;

	try {
		this->sendMetricsMethod(reportDict);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to add metrics, exception: " << e.what());
	}
}

void RLGPC::MetricSender::OnKillSignal(const int signal)
{
	RG_LOG("Received end signal " << signal << ".");
	try {
		pybind11::module pyMod = py::module::import("python_scripts.metric_receiver");
		pyMod.attr("end")(signal);
	}
	catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed during end signal handling, exception: " << e.what());
	}

}

RLGPC::MetricSender::~MetricSender() {
}