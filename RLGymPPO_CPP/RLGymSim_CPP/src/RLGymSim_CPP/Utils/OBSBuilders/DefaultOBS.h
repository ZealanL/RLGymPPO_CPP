#pragma once
#include "OBSBuilder.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/obs_builders/default_obs.py
namespace RLGSC {
	class DefaultOBS : public OBSBuilder {
	public:

		Vec posCoef;
		float velCoef, angVelCoef;
		DefaultOBS(
			Vec posCoef = Vec(1 / CommonValues::SIDE_WALL_X, 1 / CommonValues::BACK_WALL_Y, 1 / CommonValues::CEILING_Z),
			float velCoef = 1 / CommonValues::CAR_MAX_SPEED,
			float angVelCoef = 1 / CommonValues::CAR_MAX_ANG_VEL
		) : posCoef(posCoef), velCoef(velCoef), angVelCoef(angVelCoef) {

		}

		void AddPlayerToOBS(FList& obs, const PlayerData& player, bool inv);

		virtual FList BuildOBS(const PlayerData& player, const GameState& state, const Action& prevAction);
	};
}