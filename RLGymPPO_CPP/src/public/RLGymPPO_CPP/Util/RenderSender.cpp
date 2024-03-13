#include "RenderSender.h"

#include "../../libsrc/json/nlohmann/json.hpp"

namespace py = pybind11;
using namespace nlohmann;
using namespace RLGSC;

RLGPC::RenderSender::RenderSender() {
	RG_LOG("Initializing RenderSender...");

	try {
		RG_LOG("Current dir: " << std::filesystem::current_path());
		pyMod = py::module::import("python_scripts.render_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE("RenderSender: Failed to import render receiver, exception: " << e.what());
	}

	RG_LOG(" > RenderSender initalized.");
}

FList VecToList(const Vec& vec) {
	return FList({ vec.x, vec.y, vec.z });
}

json PhysToJSON(const PhysObj& obj) {
	json j = {};

	j["pos"] = VecToList(obj.pos);

	j["forward"] = VecToList(obj.rotMat.forward);
	j["right"] = VecToList(obj.rotMat.right);
	j["up"] = VecToList(obj.rotMat.up);

	j["vel"] = VecToList(obj.vel);
	j["ang_vel"] = VecToList(obj.angVel);

	return j;
}

json PlayerToJSON(const PlayerData& player) {
	json j = {};

	j["car_id"] = player.carId;
	j["team_num"] = (int)player.team;
	j["phys"] = PhysToJSON(player.phys);
	j["boost_pickups"] = player.boostPickups;
	j["is_demoed"] = player.carState.isDemoed;
	j["on_ground"] = player.carState.isOnGround;
	j["ball_touched"] = player.ballTouchedStep;
	j["has_flip"] = player.hasFlip;
	j["boost_amount"] = player.boostFraction;

	return j;
}

json GameStateToJSON(const GameState& state) {
	json j = {};
	
	j["ball"] = PhysToJSON(state.ball);

	std::vector<json> players;
	for (auto& player : state.players)
		players.push_back(PlayerToJSON(player));

	j["players"] = players;
	j["boost_pads"] = state.boostPads;
	j["team_goals"] = state.scoreLine.teamGoals;

	return j;
}

std::vector<json> ActionSetToJSON(const ActionSet& actions) {
	std::vector<json> js = {};
	for (auto& action : actions) {
		FList vals;
		for (float v : action)
			vals.push_back(v);
		js.push_back(json(vals));
	}

	return js;
}

void RLGPC::RenderSender::Send(const GameState& state, const ActionSet& actions) {
	json j = {};
	j["state"] = GameStateToJSON(state);
	j["actions"] = ActionSetToJSON(actions);
	
	std::string jStr = j.dump();

	try {
		pyMod.attr("render_state")(jStr);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("RenderSender: Failed to send gamestate, exception: " << e.what());
	}
}

RLGPC::RenderSender::~RenderSender() {

}