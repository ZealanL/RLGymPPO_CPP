#pragma once

#include <rlbot/bot.h>
#include <RLGymSim_CPP/Utils/OBSBuilders/OBSBuilder.h>
#include <RLGymSim_CPP/Utils/ActionParsers/ActionParser.h>

#include <RLGymPPO_CPP/Util/PolicyInferUnit.h>

struct RLBotParams {
	// Set this to the same port used in rlbot/port.cfg
	int port;

	RLGSC::OBSBuilder* obsBuilder = NULL; // Use your OBS builder
	RLGSC::ActionParser* actionParser = NULL; // Use your action parser

	std::filesystem::path policyPath; // The path to your trained PPO_POLICY.lt
	int obsSize; // You can find this from the console when running training
	std::vector<int> policyLayerSizes = {}; // Your layer sizes
	int tickSkip; // Your tick skip
};

class RLBotBot : public rlbot::Bot {
public:

	// Parameters to define the bot
	RLBotParams params;

	// Inference unit to infer the policy with, also uses our obs and action parser
	RLGPC::PolicyInferUnit* policyInferUnit;

	// Queued action and current action
	RLGSC::Action 
		action = {}, 
		controls = {};

	// Persistent info
	bool updateAction = true;
	float prevTime = 0;
	int ticks = -1;

	RLBotBot(int _index, int _team, std::string _name, const RLBotParams& params);
	~RLBotBot();

	rlbot::Controller GetOutput(rlbot::GameTickPacket gameTickPacket) override;
};

namespace RLBotClient {
	void Run(const RLBotParams& params);
}