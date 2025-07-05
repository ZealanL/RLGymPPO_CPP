#pragma once

#include <RLGym/Reward.h>
#include <RocketSim/RocketSim.h>

namespace RLGPC {

/**
 * @brief Reward for encouraging flip reset mechanics
 * 
 * This reward encourages the agent to position the underside of the car
 * against the ball in preparation for a flip reset. It rewards:
 * - Aligning the bottom of the car with the ball
 * - Being at appropriate height
 * - Being away from walls
 * - Facing toward the opponent's goal
 */
class FlipResetReward : public RLGSC::Reward {
public:
    /**
     * @brief Construct a new Flip Reset Reward
     * 
     * @param weight Weight of this reward component
     */
    FlipResetReward(float weight = 1.0f);

    /**
     * @brief Get the reward for the current state
     * 
     * @param state Current game state
     * @param previousActions Previous actions taken
     * @return std::vector<float> Rewards for each player
     */
    std::vector<float> GetReward(const RLGSC::GameState& state, const std::vector<RLGSC::Action>& previousActions) override;

    /**
     * @brief Reset the reward
     */
    void Reset() override;

private:
    // Constants
    static constexpr float CEILING_Z = 2044.0f;
    static constexpr float BLUE_GOAL_BACK_X = 0.0f;
    static constexpr float BLUE_GOAL_BACK_Y = -5120.0f;
    static constexpr float BLUE_GOAL_BACK_Z = 642.0f;
    static constexpr float ORANGE_GOAL_BACK_X = 0.0f;
    static constexpr float ORANGE_GOAL_BACK_Y = 5120.0f;
    static constexpr float ORANGE_GOAL_BACK_Z = 642.0f;

    // Helper methods
    float CosineSimilarity(const RS::Vec3 &v1, const RS::Vec3 &v2) const;
};

} // namespace RLGPC