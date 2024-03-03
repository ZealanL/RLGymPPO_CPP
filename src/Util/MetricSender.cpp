#include "MetricSender.h"

#include "../../libsrc/json/nlohmann/json.hpp"
#include "Timer.h"

#include <pybind11/embed.h>
namespace py = pybind11;

using namespace RLGPC;

RLGPC::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender..");

	py::initialize_interpreter();

	try {
		pyMod = py::module::import("python_scripts.metric_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to import metrics receiver, exception: " << e.what());
	}

	try {
		pyMod.attr("init")(PY_EXEC_PATH, projectName, groupName, runName);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to initialize in Python, exception: " << e.what());
	}

	RG_LOG(" > MetricSender initalized.");
}

void RLGPC::MetricSender::Send(const Report& report) {
	using namespace nlohmann;

	py::dict reportDict = {};

	for (auto& pair : report.data)
		reportDict[pair.first.c_str()] = pair.second;

	try {
		pyMod.attr("add_metrics")(reportDict);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to add metrics, exception: " << e.what());
	}
}

RLGPC::MetricSender::~MetricSender() {
	py::finalize_interpreter();
}