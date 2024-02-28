#include "Learner.h"

#include <torch/cuda.h>

RLGPC::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig config) :
	envCreateFn(envCreateFn),
	config(config), 
	device(at::Device(at::kCPU)) // Legally required to initialize this unfortunately
{
	RG_LOG("Learner::Learner():");
	
	if (config.saveFolderAddUnixTimestamp)
		config.checkpointSaveFolder += "-" + std::to_string(time(0));

	RG_LOG("\tCheckpoint Load Dir: ", config.checkpointLoadFolder);
	RG_LOG("\tCheckpoint Save Dir: ", config.checkpointSaveFolder);

	torch::manual_seed(config.randomSeed);

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
			t = t.to(device);
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
	agentMgr = new ThreadAgentManager(ppo->policy, expBuffer, config.standardizeOBS, device);

	RG_LOG("\tCreating " << config.numThreads << " agents...");
	agentMgr->CreateAgents(envCreateFn, config.numThreads, config.numGamesPerThread);
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
		"Consumption Time",
		"-PPO Learn Time",
		"--PPO Value Estimate Time",
		"--PPO Backprop Data Time",
		"--PPO Gradient Time",
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
	RG_LOG("Learner::Learn():")
	RG_LOG("\tStarting agents...");
	agentMgr->StartAgents();

	RG_LOG("\tBeginning learning loop:");
	Timer epochTimer = {};
	while (totalTimesteps < config.timestepLimit || config.timestepLimit == 0) {
		Report report = {};

		// Collect the desired timesteps from our agents
		RG_LOG("Collecting timesteps...");
;		GameTrajectory timesteps = agentMgr->CollectTimesteps(config.timestepsPerIteration);
		double collectionTime = epochTimer.Elapsed();
		uint64_t timestepsCollected = timesteps.size; // Use actual size instead of target size

		totalTimesteps += timestepsCollected;

		// Add it to our experience buffer, also computing GAE in the process
		RG_LOG("Adding experience...");
		AddNewExperience(timesteps);

		// Run the actual PPO learning on the experience we have collected
		Timer ppoLearnTimer = {};
		RG_LOG("Learning...");
		ppo->Learn(expBuffer, report);

		double ppoLearnTime = ppoLearnTimer.Elapsed();
		double epochTime = epochTimer.Elapsed();
		epochTimer.Reset(); // Reset now otherwise we can have issues with the timer and thread input-locking
		double consumptionTime = epochTime - collectionTime;

		// Get all metrics from agent manager
		agentMgr->GetMetrics(report);

		{ // Add timers to report
			report["Total Iteration Time"] = epochTime;

			report["Collection Time"] = collectionTime;
			report["Consumption Time"] = consumptionTime;
		}

		{ // Add timestep data to report
			report["Collected Steps/Second"] = (int64_t)(timestepsCollected / collectionTime);
			report["Overall Steps/Second"] = (int64_t)(timestepsCollected / epochTime);
			report["Timesteps Collected"] = timestepsCollected;
			report["Cumulative Timesteps"] = totalTimesteps;
		}

		if (iterationCallback)
			iterationCallback(report);

		{ // Print results
			constexpr const char* DIVIDER = "======================";
			RG_LOG("\n");
			RG_LOG(DIVIDER << DIVIDER);
			RG_LOG("ITERATION COMPLETED:\n");
			DisplayReport(report);
			RG_LOG(DIVIDER << DIVIDER);
			RG_LOG("\n");
		}

		// Reset everything
		agentMgr->ResetMetrics();
	}
	
	RG_LOG("Learner: Timestep limit of " << RG_COMMA_INT(config.timestepLimit) << " reached, stopping");
	RG_LOG("\tStopping agents...");
	agentMgr->StopAgents();
}

void RLGPC::Learner::AddNewExperience(GameTrajectory& gameTraj) {
	RG_NOGRAD;

	gameTraj.RemoveCapacity();
	auto& trajData = gameTraj.data;

	size_t count = trajData.actions.size(0);

	// Construct input to the value function estimator that includes the final state (which an action was not taken in)
	auto valInput = torch::cat({ trajData.states, torch::unsqueeze(trajData.nextStates[count - 1], 0) }).to(device, true);

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

	if (config.standardizeReturns) {
		int numToIncrement = RS_MIN(config.maxReturnsPerStatsInc, returns.size());
		returnStats.Increment(returns, numToIncrement);
	}

	auto expTensors = ExperienceTensors{
			trajData.states,
			trajData.actions,
			trajData.logProbs,
			trajData.rewards,
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