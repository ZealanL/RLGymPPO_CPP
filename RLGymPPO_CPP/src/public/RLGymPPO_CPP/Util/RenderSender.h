#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>
#include <RLGymSim_CPP/Utils/Gamestates/GameState.h>
#include <RLGymSim_CPP/Utils/BasicTypes/Action.h>

namespace RLGPC {
	struct RenderSender {
		pybind11::module pyMod;

		RenderSender();

		RG_NO_COPY(RenderSender);

		void Send(const RLGSC::GameState& state, const RLGSC::ActionSet& actions);

		~RenderSender();
	};
}