#include "FlipResetReward.h"
#include <cmath>
#include <algorithm>

namespace RLGPC {

FlipResetReward::FlipResetReward(float weight) : RLGSC::Reward(weight) {
    // Constructor implementation
}

std::vector<float> FlipResetReward::GetReward(const RLGSC::GameState& state, const std::vector<RLGSC::Action>& previousActions) {
    std::vector<float> rewards(state.players.size(), 0.0f);
    
    // Get ball position
    RS::Vec3 ballPos = state.ball.position;
    
    // Calculate rewards for each player
    for (size_t i = 0; i < state.players.size(); i++) {
        const auto& player = state.players[i];
        
        // Get car position and orientation
        RS::Vec3 carPos = player.position;
        RS::RotMat carRot = player.rotation;
        
        // Calculate car's up vector (negative is bottom of car)
        RS::Vec3 carUp = carRot.up;
        RS::Vec3 carUpNeg = carUp * -1.0f;
        
        // Calculate car's forward vector
        RS::Vec3 carForward = carRot.forward;
        
        // 1. Upness - bottom of car points to ceiling
        RS::Vec3 toCeiling(0, 0, CEILING_Z - carPos.z);
        float upness = CosineSimilarity(toCeiling, carUpNeg);
        
        // 2. From wall ratio - reward being away from walls
        float fromWallRatio = std::min(1.0f, std::abs(ballPos.x) / 1300.0f);
        
        // 3. Height ratio - reward being at appropriate height
        float heightRatio = std::min(1.0f, ballPos.z / 1700.0f);
        
        // 4. Bottom ball ratio - reward having bottom of car facing the ball
        RS::Vec3 toBall = ballPos - carPos;
        float bottomBallRatio = 2.0f * CosineSimilarity(toBall, carUpNeg);
        
        // 5. Align ratio - reward facing toward opponent's goal
        RS::Vec3 objective;
        if (player.team == 0) { // Blue team
            objective = RS::Vec3(ORANGE_GOAL_BACK_X, ORANGE_GOAL_BACK_Y, ORANGE_GOAL_BACK_Z);
        } else { // Orange team
            objective = RS::Vec3(BLUE_GOAL_BACK_X, BLUE_GOAL_BACK_Y, BLUE_GOAL_BACK_Z);
        }
        
        RS::Vec3 toObjective = objective - carPos;
        float alignRatio = CosineSimilarity(toObjective, carForward);
        
        // 6. Position difference - reward being close to the ball
        RS::Vec3 posDiff = ballPos - carPos;
        posDiff.z *= 2.0f; // Make z-axis twice as important
        float normPosDiff = posDiff.length();
        
        // Calculate final flip reset reward
        float flipReward = bottomBallRatio * fromWallRatio * heightRatio * alignRatio * 
                          std::clamp(40.0f * upness / (normPosDiff + 1.0f), -1.0f, 1.0f);
        
        // Add to player's reward
        rewards[i] = weight * flipReward;
    }
    
    return rewards;
}

void FlipResetReward::Reset() {
    // Nothing to reset for this reward
}

float FlipResetReward::CosineSimilarity(const RS::Vec3 &v1, const RS::Vec3 &v2) const {
    // Calculate cosine similarity between two vectors
    float dotProduct = v1.dot(v2);
    float mag1 = v1.length();
    float mag2 = v2.length();
    
    if (mag1 < 1e-5 || mag2 < 1e-5) {
        return 0.0f; // Avoid division by zero
    }
    
    return dotProduct / (mag1 * mag2);
}

} // namespace RLGPC