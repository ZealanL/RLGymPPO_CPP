#include "RLBotClient.h"

#include <rlbot/platform.h>
#include <rlbot/botmanager.h>

using namespace RLGSC;
using namespace RLGPC;

// Global variable so that we can pass params to the bot factory
// TODO: This is a lame solution
RLBotParams g_RLBotParams = {};

rlbot::Bot* BotFactory(int index, int team, std::string name) {
	return new RLBotBot(index, team, name, g_RLBotParams);
}

RLBotBot::RLBotBot(int _index, int _team, std::string _name, const RLBotParams& params) 
	: rlbot::Bot(_index, _team, _name), params(params) {

	RG_LOG("Creating RLBot bot: index " << _index << ", name: " << name << "...");

	RG_LOG(" > Loading policy from " << params.policyPath << "...");
	policyInferUnit = new PolicyInferUnit(params.obsBuilder, params.actionParser, params.policyPath, params.obsSize, params.policyLayerSizes, false);

	RG_LOG(" > Done!");
}

RLBotBot::~RLBotBot() {
	delete policyInferUnit;
}

Vec ToVec(const rlbot::flat::Vector3* rlbotVec) {
	return Vec(rlbotVec->x(), rlbotVec->y(), rlbotVec->z());
}

PhysObj ToPhysObj(const rlbot::flat::Physics* phys) {
	PhysObj obj = {};
	obj.pos = ToVec(phys->location());

	Angle ang = Angle(phys->rotation()->yaw(), phys->rotation()->pitch(), phys->rotation()->roll());
	obj.rotMat = ang.ToRotMat();

	obj.vel = ToVec(phys->velocity());
	obj.angVel = ToVec(phys->angularVelocity());

	return obj;
}

PlayerData ToPlayer(const rlbot::flat::PlayerInfo* playerInfo) {
	PlayerData pd = {};
	pd.carId = playerInfo->spawnId();

	pd.team = (Team)playerInfo->team();

	pd.phys = ToPhysObj(playerInfo->physics());
	pd.physInv = pd.phys.Invert();

	pd.boostFraction = playerInfo->boost() / 100.f;
	pd.carState.isOnGround = playerInfo->hasWheelContact();
	pd.carState.hasJumped = playerInfo->jumped();
	pd.carState.hasDoubleJumped = playerInfo->doubleJumped();
	pd.carState.isDemoed = playerInfo->isDemolished();
	pd.hasFlip = !playerInfo->doubleJumped();

	return pd;
}

GameState ToGameState(rlbot::GameTickPacket& gameTickPacket) {
	GameState gs = {};

	auto players = gameTickPacket->players();
	for (int i = 0; i < players->size(); i++)
		gs.players.push_back(ToPlayer(players->Get(i)));

	gs.ball = ToPhysObj(gameTickPacket->ball()->physics());
	gs.ballInv = gs.ball.Invert();

	auto boostPadStates = gameTickPacket->boostPadStates();
	if (boostPadStates->size() != CommonValues::BOOST_LOCATIONS_AMOUNT) {
		if (rand() % 20 == 0) { // Don't spam-log as that will lag the bot
			RG_LOG(
				"RLBotClient ToGameState(): Bad boost pad amount, expected " << CommonValues::BOOST_LOCATIONS_AMOUNT << " but got " << boostPadStates->size()
			);
		}

		// Just set all boost pads to on
		std::fill(gs.boostPads.begin(), gs.boostPads.end(), 1);
	} else {
		for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
			gs.boostPads[i] = boostPadStates->Get(i)->isActive();
			gs.boostPadsInv[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1] = gs.boostPads[i];
		}
	}

	return gs;
}

rlbot::Controller RLBotBot::GetOutput(rlbot::GameTickPacket gameTickPacket) {

	float curTime = gameTickPacket->gameInfo()->secondsElapsed();
	float deltaTime = curTime - prevTime;
	prevTime = curTime;

	int ticksElapsed = roundf(deltaTime * 120);
	ticks += ticksElapsed;

	GameState gs = ToGameState(gameTickPacket);
	auto& localPlayer = gs.players[index];

	if (updateAction) {
		updateAction = false;
		action = policyInferUnit->InferPolicySingle(localPlayer, gs, controls, true);
	}

	if (ticks >= params.tickSkip || ticks == -1) {
		// Apply new action
		controls = action;

		// Trigger action update next tick
		ticks = 0;
		updateAction = true;
	}

	auto rc = rlbot::Controller();
	{
		rc.throttle = controls.throttle;
		rc.steer = controls.steer;

		rc.pitch = controls.pitch;
		rc.yaw = controls.yaw;
		rc.roll = controls.roll;

		rc.boost = controls.boost;
		rc.jump = controls.jump;
		rc.handbrake = controls.handbrake;
	}

	return rc;
}

void RLBotClient::Run(const RLBotParams& params) {
	g_RLBotParams = params;

	rlbot::platform::SetWorkingDirectory(
		rlbot::platform::GetExecutableDirectory()
	);

	rlbot::BotManager botManager(BotFactory);
	botManager.StartBotServer(params.port);
}