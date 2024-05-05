#include "Learner.h"

#include <RLGymPPO_CPP/PPO/PPOLearner.h>
#include <RLGymPPO_CPP/PPO/ExperienceBuffer.h>
#include <RLGymPPO_CPP/Threading/ThreadAgentManager.h>

#include <torch/cuda.h>
#include "../libsrc/json/nlohmann/json.hpp"
#include <pybind11/embed.h>

RLGPC::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig _config) :
	envCreateFn(envCreateFn),
	config(_config)
{
	torch::set_num_interop_threads(1);
	torch::set_num_threads(1);

	pybind11::initialize_interpreter();

#ifndef NDEBUG
	RG_LOG("===========================");
	RG_LOG("WARNING: RLGym-PPO runs extremely slowly in debug, and there are often bizzare issues with debug-mode torch.");
	RG_LOG("It is recommended that you compile in release mode without optimization for debugging.");
	RG_SLEEP(1000);
#endif

	if (config.timestepsPerSave == 0)
		config.timestepsPerSave = config.timestepsPerIteration;

	if (config.standardizeOBS)
		RG_ERR_CLOSE("LearnerConfig.standardizeOBS has not yet been implemented, sorry");

	RG_LOG("Learner::Learner():");
	
	if (config.renderMode) {
		RG_LOG("\tRender mode is enabled, overriding:");
		config.numThreads = config.numGamesPerThread = 1;
		RG_LOG("\t > numThreads, numGamesPerThread = 1");
		
		config.sendMetrics = false;
		RG_LOG("\t > sendMetrics = false");

		config.checkpointSaveFolder.clear();
		RG_LOG("\t > checkpointSaveFolder = none");

		config.timestepsPerIteration = INT_MAX;
		RG_LOG("\t > timestepsPerIteration = inf");
	}

	if (config.saveFolderAddUnixTimestamp && !config.checkpointSaveFolder.empty())
		config.checkpointSaveFolder += "-" + std::to_string(time(0));

	RG_LOG("\tCheckpoint Load Dir: " << config.checkpointLoadFolder);
	RG_LOG("\tCheckpoint Save Dir: " << config.checkpointSaveFolder);

	torch::manual_seed(config.randomSeed);

	at::Device device = at::Device(at::kCPU);
	if (
		config.deviceType == LearnerDeviceType::GPU_CUDA || 
		(config.deviceType == LearnerDeviceType::AUTO && torch::cuda::is_available())
		) {
		RG_LOG("\tUsing CUDA GPU device...");

		// Test out moving a tensor to GPU and back to make sure the device is working
		torch::Tensor t;
		bool deviceTestFailed = false;
		try {
			t = torch::tensor(0);
			t = t.to(at::Device(at::kCUDA));
			t = t.cpu();
		} catch (...) {
			deviceTestFailed = true;
		}

		if (!torch::cuda::is_available() || deviceTestFailed)
			RG_ERR_CLOSE(
				"Learner::Learner(): Can't use CUDA GPU because " <<
				(torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
			)
		device = at::Device(at::kCUDA);
	} else {
		RG_LOG("\tUsing CPU device...");
		device = at::Device(at::kCPU);
	}

	if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
		RG_LOG("\tInitializing RocketSim...");
		RocketSim::Init("collision_meshes");
	}

	{
		RG_LOG("\tCreating test environment to determine OBS size and action amount...")
		auto envCreateResult = envCreateFn();
		auto obsSet = envCreateResult.gym->Reset();
		obsSize = obsSet[0].size();
		actionAmount = envCreateResult.match->actionParser->GetActionAmount();
		RG_LOG("\t\tOBS size: " << obsSize);
		RG_LOG("\t\tAction amount: " << actionAmount);
		delete envCreateResult.gym;
		delete envCreateResult.match;
	}

	RG_LOG("\tCreating experience buffer...");
	expBuffer = new ExperienceBuffer(config.expBufferSize, config.randomSeed, device);

	RG_LOG("\tCreating PPO Learner...");
	ppo = new PPOLearner(obsSize, actionAmount, config.ppo, device);

	RG_LOG("\tCreating agent manager...");
	agentMgr = new ThreadAgentManager(
		ppo->policy, ppo->policyHalf, expBuffer, 
		config.standardizeOBS, config.deterministic,
		(uint64_t)(config.timestepsPerIteration * 1.5f),
		device
	);

	RG_LOG("\tCreating " << config.numThreads << " agents...");
	agentMgr->CreateAgents(envCreateFn, config.numThreads, config.numGamesPerThread);

	if (!config.checkpointLoadFolder.empty())
		Load();

	if (config.sendMetrics) {
		metricSender = new MetricSender(config.metricsProjectName, config.metricsGroupName, config.metricsRunName, runID);
	} else {
		metricSender = NULL;
	}

	if (config.renderMode) {
		renderSender = new RenderSender();
		agentMgr->renderSender = renderSender;
		agentMgr->renderTimeScale = config.renderTimeScale;
	} else {
		renderSender = NULL;
	}
}

template <typename T>
nlohmann::json MakeJSONArray(const std::vector<T> list) {
	auto result = nlohmann::json::array();

	for (T v : list) {
		if (isnan(v))
			RG_LOG("MakeJSONArray(): Failed to serialize JSON with NAN value (list size: " << list.size() << ")");
		result.push_back(v);
	}

	return result;
}

void RLGPC::Learner::SaveStats(std::filesystem::path path) {
	using namespace nlohmann;

	constexpr const char* ERROR_PREFIX = "Learner::SaveStats(): ";

	std::ofstream fOut(path);
	if (!fOut.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = {};
	j["cumulative_timesteps"] = totalTimesteps;
	j["cumulative_model_updates"] = ppo->cumulativeModelUpdates;
	j["epoch"] = totalEpochs;
	
	auto& rrs = j["reward_running_stats"];
	{
		rrs["mean"] = MakeJSONArray(returnStats.runningMean);
		rrs["var"] = MakeJSONArray(returnStats.runningVariance);
		rrs["shape"] = returnStats.shape;
		rrs["count"] = returnStats.count;
	}

	if (config.sendMetrics)
		j["run_id"] = metricSender->curRunID;

	std::string jStr = j.dump(4);
	fOut << jStr;
}

void RLGPC::Learner::LoadStats(std::filesystem::path path) {
	// TODO: Repetitive code, merge repeated code into one function called from both SaveStats() and LoadStats()

	using namespace nlohmann;
	constexpr const char* ERROR_PREFIX = "Learner::LoadStats(): ";

	std::ifstream fIn(path);
	if (!fIn.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = json::parse(fIn);
	totalTimesteps = j["cumulative_timesteps"];
	ppo->cumulativeModelUpdates = j["cumulative_model_updates"];
	totalEpochs = j["epoch"];
	
	auto& rrs = j["reward_running_stats"];
	{
		returnStats = WelfordRunningStat(rrs["shape"]);
		returnStats.runningMean = rrs["mean"].get<std::vector<double>>();
		returnStats.runningVariance = rrs["var"].get<std::vector<double>>();
		returnStats.count = rrs["count"];
	}

	if (j.contains("run_id"))
		runID = j["run_id"];
}

// Different than RLGym-PPO to show that they are not compatible
constexpr const char* STATS_FILE_NAME = "RUNNING_STATS.json";

void RLGPC::Learner::Save() {
	if (config.checkpointSaveFolder.empty())
		RG_ERR_CLOSE("Learner::Save(): Cannot save because config.checkpointSaveFolder is not set");

	std::filesystem::path saveFolder = config.checkpointSaveFolder / std::to_string(totalTimesteps);
	std::filesystem::create_directories(saveFolder);

	RG_LOG("Saving to folder " << saveFolder << "...");
	SaveStats(saveFolder / STATS_FILE_NAME);
	ppo->SaveTo(saveFolder);

	// Remove old checkpoints
	if (config.checkpointsToKeep != -1) {
		int numCheckpoints = 0;
		int64_t lowestCheckpointTS = INT64_MAX;

		for (auto entry : std::filesystem::directory_iterator(config.checkpointLoadFolder)) {
			if (entry.is_directory()) {
				auto name = entry.path().filename();
				try {
					int64_t nameVal = std::stoll(name);
					lowestCheckpointTS = RS_MIN(nameVal, lowestCheckpointTS);
					numCheckpoints++;
				} catch (...) {}
			}
		}

		if (numCheckpoints > config.checkpointsToKeep) {
			std::filesystem::path removePath = config.checkpointLoadFolder / std::to_string(lowestCheckpointTS);
			try {
				std::filesystem::remove_all(removePath);
			} catch (std::exception& e) {
				RG_ERR_CLOSE("Failed to remove old checkpoint from " << removePath << ", exception: " << e.what());
			}
		}
	}

	RG_LOG(" > Done.");
}

void RLGPC::Learner::Load() {
	if (config.checkpointLoadFolder.empty())
		RG_ERR_CLOSE("Learner::Load(): Cannot load because config.checkpointLoadFolder is not set");

	RG_LOG("Loading most recent checkpoint in " << config.checkpointLoadFolder << "...");

	int64_t highest = -1;
	if (std::filesystem::is_directory(config.checkpointLoadFolder)) {
		for (auto entry : std::filesystem::directory_iterator(config.checkpointLoadFolder)) {
			if (entry.is_directory()) {
				auto name = entry.path().filename();
				try {
					int64_t nameVal = std::stoll(name);
					highest = RS_MAX(nameVal, highest);
				} catch (...) {}
			}
		}
	}

	if (highest != -1) {
		std::filesystem::path loadFolder = config.checkpointLoadFolder / std::to_string(highest);
		RG_LOG(" > Loading checkpoint " << loadFolder << "...");
		LoadStats(loadFolder / STATS_FILE_NAME);
		ppo->LoadFrom(loadFolder);
		RG_LOG(" > Done.");
	} else {
		RG_LOG(" > No checkpoints found, starting new model.")
	}
}

// Prints the metrics report in a similar way to rlgym-ppo
void DisplayReport(const RLGPC::Report& report) {
	// FORMAT:
	//	blank line = print blank line
	//	'-' before name = indent with dashes and spaces
	constexpr const char* REPORT_DATA_ORDER[] = {
		"Average Episode Reward",
		"Average Step Reward",
		"Policy Entropy",
		"Value Function Loss",
		"",
		"Mean KL Divergence",
		"SB3 Clip Fraction",
		"Policy Update Magnitude",
		"Value Function Update Magnitude",
		"",
		"Collected Steps/Second",
		"Overall Steps/Second",
		"",
		"Collection Time",
		"-Policy Infer Time",
		"-Env Step Time",
		"Consumption Time",
		"-PPO Learn Time",
		"Collect-Consume Overlap Time",
		// TODO: These timers don't work due to non-blocking mode
		//"--PPO Value Estimate Time",
		//"--PPO Backprop Data Time",
		//"--PPO Gradient Time",
		"Total Iteration Time",
		"",
		"Cumulative Model Updates",
		"Cumulative Timesteps",
		"",
		"Timesteps Collected"
	};

	for (const char* name : REPORT_DATA_ORDER) {
		if (strlen(name) > 0) {
			int indentLevel = 0;
			while (name[0] == '-') {
				indentLevel++;
				name++;
			}

			std::string prefix = {};
			if (indentLevel > 0) {
				prefix += std::string((indentLevel - 1) * 3, ' ');
				prefix += " - ";
			}

			RG_LOG(prefix << report.SingleToString(name, true));
		} else {
			RG_LOG("");
		}
	}
}

void RLGPC::Learner::Learn() {
	RG_LOG("Learner::Learn():");

#ifdef RG_PARANOID_MODE
	RG_LOG("NOTE: Paranoid mode active. Additional checks will be run that may impact performance.");
#endif

	RG_LOG("\tStarting agents...");
	agentMgr->SetStepCallback(stepCallback);
	agentMgr->StartAgents();

	auto device = ppo->device;

	RG_LOG("\tBeginning learning loop:");
	int64_t tsSinceSave = 0;
	Timer epochTimer = {};
	while (totalTimesteps < config.timestepLimit || config.timestepLimit == 0) {
		Report report = {};

		agentMgr->SetStepCallback(stepCallback);

		// Collect the desired timesteps from our agents
		GameTrajectory timesteps = agentMgr->CollectTimesteps(config.timestepsPerIteration);
		double relCollectionTime = epochTimer.Elapsed();
		uint64_t timestepsCollected = timesteps.size; // Use actual size instead of target size

		totalTimesteps += timestepsCollected;

		if (!config.collectionDuringLearn)
			agentMgr->disableCollection = true;

		// Add it to our experience buffer, also computing GAE in the process
		try {
			AddNewExperience(timesteps, report);
		} catch (std::exception& e) {
			RG_ERR_CLOSE("Exception during Learner::AddNewExperience(): " << e.what());
		}

		Timer ppoLearnTimer = {};

		// Stop agents from inferencing during learn if we are not on CPU
		// This is because learning is very GPU intensive, and letting iterations collect during that time slows it down
		// On CPU, learning is its own thread, it's better to keep collecting
		// Also, if config.collectionDuringLearn is false, we ignore this
		bool blockAgentInferDuringLearn = config.collectionDuringLearn && !device.is_cpu();
		{ // Run the actual PPO learning on the experience we have collected
			
			if (config.deterministic) {
				RG_ERR_CLOSE(
					"Learner::Learn(): Cannot run PPO learn iteration when on deterministic mode!"
					"\nDeterministic mode is meant for performing, not training. Only collection should occur."
				);
			}

			RG_LOG("Learning...");
			if (blockAgentInferDuringLearn)
				agentMgr->disableCollection = true;

			try {
				ppo->Learn(expBuffer, report);
			} catch (std::exception& e) {
				RG_ERR_CLOSE("Exception during PPOLearner::Learn(): " << e.what());
			}

			if (blockAgentInferDuringLearn)
				agentMgr->disableCollection = false;

			totalEpochs += config.ppo.epochs;
		}

		double ppoLearnTime = ppoLearnTimer.Elapsed();
		double relEpochTime = epochTimer.Elapsed();
		epochTimer.Reset(); // Reset now otherwise we can have issues with the timer and thread input-locking

		double consumptionTime = relEpochTime - relCollectionTime;

		// Get all metrics from agent manager
		agentMgr->GetMetrics(report);

		if (!config.collectionDuringLearn) {
			agentMgr->disableCollection = false;
		}

		// If we collect during consuption, don't just measure the time we waited for to collect for steps
		// Because of collection during learn, this time could be near-zero, resulting in SPS showing some crazy number
		double trueCollectionTime = config.collectionDuringLearn ? agentMgr->lastIterationTime : relCollectionTime;
		if (blockAgentInferDuringLearn)
			trueCollectionTime -= ppoLearnTime; // We couldn't have been collecting during this time
		trueCollectionTime = RS_MAX(trueCollectionTime, relCollectionTime);

		// Fix same issue with epoch time
		double trueEpochTime = RS_MAX(relEpochTime, trueCollectionTime);

		{ // Add timers to report
			report["Total Iteration Time"] = relEpochTime;

			report["Collection Time"] = relCollectionTime;
			report["Consumption Time"] = consumptionTime;
			report["Collect-Consume Overlap Time"] = (trueCollectionTime - relCollectionTime);
		}

		{ // Add timestep data to report
			report["Collected Steps/Second"] = (int64_t)(timestepsCollected / trueCollectionTime);
			report["Overall Steps/Second"] = (int64_t)(timestepsCollected / trueEpochTime);
			report["Timesteps Collected"] = timestepsCollected;
			report["Cumulative Timesteps"] = totalTimesteps;
		}

		// Call iteration callback
		if (iterationCallback) {
			RG_LOG("Calling iteration callback...");
			iterationCallback(this, report);
		}

		{ // Print results
			constexpr const char* DIVIDER = "======================";
			RG_LOG("\n");
			RG_LOG(DIVIDER << DIVIDER);
			RG_LOG("ITERATION COMPLETED:\n");
			DisplayReport(report);
			RG_LOG(DIVIDER << DIVIDER);
			RG_LOG("\n");
		}

		// Update metric sender
		if (config.sendMetrics)
			metricSender->Send(report);

		// Save if needed
		tsSinceSave += timestepsCollected;
		if (tsSinceSave > config.timestepsPerSave && !config.checkpointSaveFolder.empty()) {
			Save();
			tsSinceSave = 0;
		}

		// Reset everything
		agentMgr->ResetMetrics();
	}
	
	RG_LOG("Learner: Timestep limit of " << config.timestepLimit << " reached, stopping");
	RG_LOG("\tStopping agents...");
	agentMgr->StopAgents();
}

void RLGPC::Learner::AddNewExperience(GameTrajectory& gameTraj, Report& report) {
	RG_NOGRAD;

	RG_LOG("Adding experience...");

	gameTraj.RemoveCapacity();
	auto& trajData = gameTraj.data;

	size_t count = trajData.actions.size(0);

	// Construct input to the value function estimator that includes the final state (which an action was not taken in)
	auto valInput = 
		torch::cat({ trajData.states, torch::unsqueeze(trajData.nextStates[count - 1], 0) })
		.to(ppo->device, true);

	auto valPredsTensor = ppo->valueNet->Forward(valInput).cpu().flatten();
	FList valPreds = TENSOR_TO_FLIST(valPredsTensor);
	// TODO: rlgym-ppo runs torch.cuda.empty_cache() here
	
	float retStd = (config.standardizeReturns ? returnStats.GetSTD()[0] : 1);

	// Compute GAE stuff
	torch::Tensor advantages, valueTargets;
	FList returns;
	TorchFuncs::ComputeGAE(
		TENSOR_TO_FLIST(trajData.rewards),
		TENSOR_TO_FLIST(trajData.dones),
		TENSOR_TO_FLIST(trajData.truncateds),
		valPreds,
		advantages,
		valueTargets,
		returns,
		config.gaeGamma,
		config.gaeLambda,
		retStd
	);

	float avgRet = 0;
	for (float f : returns)
		avgRet += abs(f);
	avgRet /= returns.size();
	report["Avg Return"] = avgRet / retStd;

	if (config.standardizeReturns) {
		int numToIncrement = RS_MIN(config.maxReturnsPerStatsInc, returns.size());
		returnStats.Increment(returns, numToIncrement);
	}

	auto expTensors = ExperienceTensors{
			trajData.states,
			trajData.actions,
			trajData.logProbs,
			trajData.rewards,

#ifdef RG_PARANOID_MODE
			trajData.debugCounters,
#endif

			trajData.nextStates,
			trajData.dones,
			trajData.truncateds,
			valueTargets,
			advantages
	};
	expBuffer->SubmitExperience(
		expTensors
	);
}

void RLGPC::Learner::UpdateLearningRates(float policyLR, float criticLR) {
	ppo->UpdateLearningRates(policyLR, criticLR);
}

std::vector<RLGPC::Report> RLGPC::Learner::GetAllGameMetrics() {
	std::vector<Report> reports = {};

	for (auto agent : agentMgr->agents) {
		agent->gameStepMutex.lock();
		for (auto game : agent->gameInsts)
			reports.push_back(game->_metrics);
		agent->gameStepMutex.unlock();
	}

	return reports;
}

RLGPC::Learner::~Learner() {
	delete ppo;
	delete agentMgr;
	delete expBuffer;
	delete metricSender;
	delete renderSender;
	pybind11::finalize_interpreter();
}