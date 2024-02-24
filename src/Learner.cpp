#include "Learner.h"

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

void RLGPC::Learner::Learn() {
	RG_LOG("Learner::Learn():")
	RG_LOG("\tStarting agents...");
	agentMgr->StartAgents();

	RG_LOG("\tBeginning learning loop:");
	Timer epochTimer = {};
	while (totalTimesteps < config.timestepLimit || config.timestepLimit == 0) {
		// Collect the desired timesteps from our agents
		RG_LOG("Collecting timesteps...");
;		GameTrajectory timesteps = agentMgr->CollectTimesteps(config.timestepsPerIteration);
		auto totalAgentTimes = agentMgr->GetTotalAgentTimes();
		double collectionTime = epochTimer.Elapsed();
		uint64_t timestepsCollected = timesteps.size; // Use actual size instead of target size

		totalTimesteps += timestepsCollected;

		// Add it to our experience buffer, also computing GAE in the process
		RG_LOG("Adding experience...");
		AddNewExperience(timesteps);

		// Run the actual PPO learning on the experience we have collected
		Timer ppoLearnTimer = {};
		RG_LOG("Learning...");
		auto metrics = ppo->Learn(expBuffer);

		double ppoLearnTime = ppoLearnTimer.Elapsed();
		double epochTime = epochTimer.Elapsed();
		epochTimer.Reset(); // Reset now otherwise we can have issues with the timer

		{ // Print results
			constexpr const char* DIVIDER = "======================";
			std::stringstream msg;
			msg << "\n\n\n\n"; // Make some space
			msg << DIVIDER << DIVIDER << std::endl;
			msg << " ITERATION COMPLETED:" << std::endl;
			msg << " Metrics:" << std::endl;
			msg << metrics.ToString(" - ");
			msg << DIVIDER << std::endl;
			msg << " Average reward (per tick): " << agentMgr->GetAvgReward() << std::endl;
			msg << DIVIDER << std::endl;
			msg << " Total iteration time: " << epochTime << "s" << std::endl;
			msg << " -  Collection:        " << collectionTime << "s" << std::endl;
			msg << "    - Env step:        " << totalAgentTimes.envStepTime << "s" << std::endl;
			msg << "    - Policy infer:    " << totalAgentTimes.policyInferTime << "s" << std::endl;
			msg << "    - Traj append:     " << totalAgentTimes.trajAppendTime << "s" << std::endl;
			
			msg << " - Consumption time:   " << (epochTime - collectionTime) << "s" << std::endl;
			msg << "    - PPO learn time:  " << ppoLearnTime << "s" << std::endl;
			msg << " Collected Steps/Second: " << RG_COMMA_INT(timestepsCollected / collectionTime) << std::endl;
			msg << "   Overall Steps/Second: " << RG_COMMA_INT(timestepsCollected / epochTime) << std::endl;
			msg << DIVIDER << std::endl;
			msg << " Timesteps collected: " << RG_COMMA_INT(timestepsCollected) << std::endl;
			msg << " Cumulative timesteps: " << RG_COMMA_INT(totalTimesteps) << std::endl;
			RG_LOG(msg.str());
		}

		agentMgr->ResetAvgReward();
		agentMgr->ResetAgentTimes();
	}

	RG_LOG("Learner: Timestep limit of " << config.timestepLimit << " reached, stopping");
	RG_LOG("\tStopping agents...");
	agentMgr->StopAgents();
}

void RLGPC::Learner::AddNewExperience(GameTrajectory& gameTraj) {
	RG_NOGRAD;

	gameTraj.RemoveCapacity();
	auto& trajData = gameTraj.data;

	size_t count = trajData.actions.size(0);

	// Construct input to the value function estimator that includes the final state (which an action was not taken in)
	auto valInput = torch::cat({ trajData.states, torch::unsqueeze(trajData.nextStates[count - 1], 0) }).to(device);

	auto valPredsTensor = ppo->valueNet->Forward(valInput).cpu().flatten();
	FList valPreds = TENSOR_TO_FLIST(valPredsTensor);
	// TODO: rlgym-ppo runs torch.cuda.empty_cache() here
	
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
		config.gaeLambda
	);

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