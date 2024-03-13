#pragma once
#include "RewardFunction.h"

namespace RLGSC {
	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/misc_rewards.py
	class EventReward : public RewardFunction {
	public:
		struct ValSet {
			constexpr static int VAL_AMOUNT = 9;
			float vals[VAL_AMOUNT] = {};

			float& operator[](int index) { return vals[index]; }
		};
		ValSet weights;

		std::unordered_map<int, ValSet> lastRegisteredValues;

		struct WeightScales {
			float
				goal = 0,
				teamGoal = 0,
				concede = 0,

				touch = 0,
				shot = 0,
				save = 0,
				demo = 0,
				demoed = 0,
				boostPickup = 0;

			float& operator[](size_t index) { 
				// Make sure members line up
				static_assert(
					offsetof(WeightScales, boostPickup) - offsetof(WeightScales, goal) == 
					sizeof(float) * (ValSet::VAL_AMOUNT - 1)
					);
				return (&goal)[index]; 
			}
		};

		EventReward(WeightScales scales);

		static ValSet ExtractValues(const PlayerData& player, const GameState& state);

		virtual void Reset(const GameState& state);
		virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction);
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/misc_rewards.py
	class VelocityReward : public RewardFunction {
	public:
		bool isNegative;
		VelocityReward(bool isNegative = false) : isNegative(isNegative) {}
		virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
			return player.phys.vel.Length() / CommonValues::CAR_MAX_SPEED * (1 - 2 * isNegative);
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/misc_rewards.py
	class SaveBoostReward : public RewardFunction {
	public:
		float exponent;
		SaveBoostReward(float exponent = 0.5f) : exponent(exponent) {}

		virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
			return RS_CLAMP(powf(player.boostFraction, exponent), 0, 1);
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/ball_goal_rewards.py
	class VelocityBallToGoalReward : public RewardFunction {
	public:
		bool ownGoal = false;
		VelocityBallToGoalReward(bool ownGoal = false) : ownGoal(ownGoal) {}

		virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
			bool targetOrangeGoal = player.team == Team::BLUE;
			if (ownGoal)
				targetOrangeGoal = !targetOrangeGoal;

			Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			
			Vec ballDirToGoal = (targetPos - state.ball.pos).Normalized();
			return ballDirToGoal.Dot(state.ball.vel / CommonValues::BALL_MAX_SPEED);
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/player_ball_rewards.py
	class VelocityPlayerToBallReward : public RewardFunction {
	public:
		virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
			Vec dirToBall = (state.ball.pos - player.phys.pos).Normalized();
			Vec normVel = player.phys.vel / CommonValues::CAR_MAX_SPEED;
			return dirToBall.Dot(normVel);
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/player_ball_rewards.py
	class FaceBallReward : public RewardFunction {
	public:
		virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
			Vec dirToBall = (state.ball.pos - player.phys.pos).Normalized();
			return player.carState.rotMat.forward.Dot(dirToBall);
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/player_ball_rewards.py
	class TouchBallReward : public RewardFunction {
	public:
		float aerialWeight;
		TouchBallReward(float aerialWeight = 0) : aerialWeight(aerialWeight) {}

		virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
			using namespace CommonValues;

			if (player.ballTouchedStep) {
				return powf((state.ball.pos.z + BALL_RADIUS) / (BALL_RADIUS * 2), aerialWeight);
			} else {
				return 0;
			}
		}
	};
}