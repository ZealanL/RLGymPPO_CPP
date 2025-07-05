#include <RLGym/Reward.h>
#include <RLGym/RewardFunctions/CombinedReward.h>
#include <RLGymPPO_CPP/Rewards/FlipResetReward.h>
#include <RLGym/RewardFunctions/CommonRewards.h>

// Example of how to use the FlipResetReward in your reward function

RLGSC::Reward* CreateRewardFunction() {
    // Create a combined reward with multiple components
    auto combinedReward = new RLGSC::CombinedReward();
    
    // Add standard rewards
    combinedReward->AddReward(new RLGSC::VelocityPlayerToBallReward(1.0f));  // Encourage moving toward ball
    combinedReward->AddReward(new RLGSC::TouchBallReward(5.0f));             // Encourage touching the ball
    combinedReward->AddReward(new RLGSC::TeamGoalReward(10.0f));             // Reward scoring goals
    
    // Add the flip reset reward with a weight of 2.0
    // Adjust this weight based on how much you want to encourage flip resets
    combinedReward->AddReward(new RLGPC::FlipResetReward(2.0f));
    
    return combinedReward;
}

// In your main code, you would use this reward function when creating your environment:
/*
auto match = new RLGSC::Match(
    // ... other parameters ...
    CreateRewardFunction()
);
*/