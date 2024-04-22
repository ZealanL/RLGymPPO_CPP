#pragma once

#include <rlbot/bot.h>
#include <RLGymSim_CPP/Utils/OBSBuilders/OBSBuilder.h>
#include <RLGymSim_CPP/Utils/ActionParsers/ActionParser.h>

#include <RLGymPPO_CPP/Util/PolicyInferUnit.h>

struct RLBotParams {
	int port;

	RLGSC::OBSBuilder* obsBuilder = NULL;
	RLGSC::ActionParser* actionParser = NULL;

	std::filesystem::path policyPath;
	int obsSize;
	std::vector<int> policyLayerSizes = {};
	int tickSkip;
};

class RLBotBot : public rlbot::Bot {
public:

	RLBotParams params;

	RLGPC::PolicyInferUnit* policyInferUnit;

	RLGSC::Action 
		action = {}, 
		controls = {};

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