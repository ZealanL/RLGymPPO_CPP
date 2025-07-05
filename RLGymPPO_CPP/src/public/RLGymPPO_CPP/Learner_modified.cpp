#include "Learner.h"
#include "Learner.h"

#include "../../private/RLGymPPO_CPP/Util/SkillTracker.h"
#include "../../private/RLGymPPO_CPP/Util/PolicyPool.h"

#include <RLGymPPO_CPP/PPO/PPOLearner.h>
#include <RLGymPPO_CPP/PPO/ExperienceBuffer.h>
#include <RLGymPPO_CPP/Threading/ThreadAgentManager.h>

#include <torch/cuda.h>
#include "../libsrc/json/nlohmann/json.hpp"
#include <pybind11/embed.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#endif

RLGPC::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig _config) :
	envCreateFn(envCreateFn),
	config(_config)
{
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

	if (config.renderMode && !config.renderDuringTraining) {
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

	torch::set_num_interop_threads(1);
	torch::set_num_threads(1);

	if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
		RG_LOG("\tInitializing RocketSim...");
		RocketSim::Init("collision_meshes", true);
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

	// Initialize policy pool if enabled
	if (config.enablePolicyPool) {
		RG_LOG("\tCreating policy pool with size " << config.policyPoolSize << " and self-play ratio " << config.selfPlayRatio);
		policyPool = new PolicyPool(config.policyPoolSize, config.selfPlayRatio, config.randomSeed);
	} else {
		policyPool = nullptr;
	}

	RG_LOG("\tCreating agent manager...");
	agentMgr = new ThreadAgentManager(
		ppo->policy, ppo->policyHalf, expBuffer, 
		config.standardizeOBS, config.deterministic, device.is_cpu() && torch::get_num_threads() > 1,
		(uint64_t)(config.timestepsPerIteration * 1.5f),
		device
	);
	
	// Set the policy pool in the agent manager
	if (policyPool) {
		agentMgr->SetPolicyPool(policyPool);
	}

	RG_LOG("\tCreating " << config.numThreads << " agents...");
	agentMgr->CreateAgents(envCreateFn, config.numThreads, config.numGamesPerThread);

	if (config.renderMode) {
		renderSender = new RenderSender();
		agentMgr->renderSender = renderSender;
		agentMgr->renderTimeScale = config.renderTimeScale;
		agentMgr->renderDuringTraining = config.renderDuringTraining;
	} else {
		renderSender = NULL;
	}

	if (config.skillTrackerConfig.enabled) {
		if (config.skillTrackerConfig.envCreateFunc == NULL)
			config.skillTrackerConfig.envCreateFunc = envCreateFn;

		skillTracker = new SkillTracker(config.skillTrackerConfig, renderSender);
	} else {
		skillTracker = NULL;
	}

	if (!config.checkpointLoadFolder.empty())
		Load();

	if (config.sendMetrics) {
		if (!runID.empty())
			RG_LOG("\tRun ID: " << runID);
		metricSender = new MetricSender(config.metricsProjectName, config.metricsGroupName, config.metricsRunName, runID);
	} else {
		metricSender = NULL;
	}
}

// Add the UpdatePolicyPool method
void RLGPC::Learner::UpdatePolicyPool() {
	if (!config.enablePolicyPool || !policyPool) {
		return;
	}

	// Add the current policy to the pool
	RG_LOG("Adding current policy to policy pool at timestep " << totalTimesteps);
	policyPool->AddPolicy(ppo->policy, totalTimesteps);
	timestepsSincePoolUpdate = 0;
}

// Modify the Learn method to update the policy pool
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
		timestepsSincePoolUpdate += timestepsCollected;

		// Check if we should update the policy pool
		if (config.enablePolicyPool && timestepsSincePoolUpdate >= config.timestepsPerPoolUpdate) {
			UpdatePolicyPool();
		}

		if (config.ppo.policyLR == 0 && config.ppo.criticLR == 0) {
			RG_LOG("\tBoth LRs are set to zero. Skipping consumption!");
#ifdef RG_CUDA_SUPPORT
			if (ppo->device.is_cuda())
				c10::cuda::CUDACachingAllocator::emptyCache();
#endif
			continue;
		}

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

		// Free CUDA cache
#ifdef RG_CUDA_SUPPORT
		if (ppo->device.is_cuda())
			c10::cuda::CUDACachingAllocator::emptyCache();
#endif

		double ppoLearnTime = ppoLearnTimer.Elapsed();
		double relEpochTime = epochTimer.Elapsed();
		epochTimer.Reset(); // Reset now otherwise we can have issues with the timer and thread input-locking

		// Update our metrics
		report.Set("Cumulative Timesteps", (double)totalTimesteps);
		report.Set("Cumulative Model Updates", (double)ppo->cumulativeModelUpdates);
		report.Set("Timesteps Collected", (double)timestepsCollected);
		report.Set("PPO Learn Time", ppoLearnTime);
		report.Set("Collection Time", relCollectionTime);
		report.Set("Collect-Consume Overlap Time", RS_MAX(0, relCollectionTime - ppoLearnTime));
		report.Set("Total Iteration Time", relEpochTime);
		report.Set("Collected Steps/Second", timestepsCollected / relCollectionTime);
		report.Set("Overall Steps/Second", timestepsCollected / relEpochTime);

		// Get metrics from our agents
		agentMgr->GetMetrics(report);
		agentMgr->ResetMetrics();

		// Display the report
		DisplayReport(report);

		// Send metrics to the python metrics receiver
		if (metricSender)
			metricSender->Send(report);

		// Run the iteration callback
		if (iterationCallback)
			iterationCallback(this, report);

		// Save if needed
		tsSinceSave += timestepsCollected;
		if (!config.checkpointSaveFolder.empty() && tsSinceSave >= config.timestepsPerSave) {
			Save();
			tsSinceSave = 0;
		}

		// Run skill tracker if enabled
		if (skillTracker && totalTimesteps > 0) {
			if (totalTimesteps % (config.skillTrackerConfig.updateInterval * config.timestepsPerIteration) == 0) {
				skillTracker->RunGames(ppo->policy, timestepsCollected);
			}

			if (config.skillTrackerConfig.startWithVersion && timestepsSincePoolUpdate >= config.skillTrackerConfig.timestepsPerVersion) {
				skillTracker->AppendOldPolicy(
					new DiscretePolicy(*ppo->policy),
					skillTracker->curRating
				);
				timestepsSincePoolUpdate = 0;
			}
		}
	}

	RG_LOG("Learner::Learn(): Finished learning loop.");
}

RLGPC::Learner::~Learner() {
	if (policyPool) {
		delete policyPool;
	}
	
	if (skillTracker)
		delete skillTracker;

	if (metricSender)
		delete metricSender;

	if (renderSender)
		delete renderSender;

	delete agentMgr;
	delete expBuffer;
	delete ppo;

	pybind11::finalize_interpreter();
}