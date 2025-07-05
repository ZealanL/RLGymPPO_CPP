# Flip Reset Reward for RLGymPPO_CPP

This reward encourages your agent to perform flip reset mechanics in Rocket League. It rewards the agent for positioning the underside of the car against the ball in preparation for a flip reset.

## What is a Flip Reset?

A flip reset is an advanced mechanic in Rocket League where a player positions their car so that all four wheels touch the ball simultaneously, which resets their flip ability. This allows for an additional flip/dodge even after being in the air for more than 1.5 seconds.

## How the Reward Works

The reward function evaluates several factors:

1. **Bottom-Ball Alignment**: Rewards having the bottom of the car facing toward the ball
2. **Wall Distance**: Rewards being away from walls (as flip resets are typically performed mid-field)
3. **Height**: Rewards being at an appropriate height
4. **Goal Alignment**: Rewards facing toward the opponent's goal
5. **Proximity**: Rewards being close to the ball
6. **Car Orientation**: Rewards having the bottom of the car pointing upward (toward the ceiling)

These factors are combined to create a comprehensive reward that guides the agent toward positioning for a flip reset.

## Integration

### 1. Add the Files to Your Project

Make sure the following files are in your project:
- `RLGymPPO_CPP/src/private/RLGymPPO_CPP/Rewards/FlipResetReward.h`
- `RLGymPPO_CPP/src/private/RLGymPPO_CPP/Rewards/FlipResetReward.cpp`
- `RLGymPPO_CPP/src/public/RLGymPPO_CPP/Rewards/FlipResetReward.h`

### 2. Include the Header

In your code, include the reward header:

```cpp
#include <RLGymPPO_CPP/Rewards/FlipResetReward.h>
```

### 3. Add to Your Reward Function

Create a combined reward that includes the flip reset reward:

```cpp
RLGSC::Reward* CreateRewardFunction() {
    auto combinedReward = new RLGSC::CombinedReward();
    
    // Add standard rewards
    combinedReward->AddReward(new RLGSC::VelocityPlayerToBallReward(1.0f));
    combinedReward->AddReward(new RLGSC::TouchBallReward(5.0f));
    combinedReward->AddReward(new RLGSC::TeamGoalReward(10.0f));
    
    // Add the flip reset reward
    // Adjust the weight (2.0f) based on how much you want to encourage flip resets
    combinedReward->AddReward(new RLGPC::FlipResetReward(2.0f));
    
    return combinedReward;
}
```

### 4. Use in Your Match

Use your reward function when creating your match:

```cpp
auto match = new RLGSC::Match(
    // ... other parameters ...
    CreateRewardFunction()
);
```

## Tuning

You may need to tune the reward to get the best results:

1. **Weight**: Adjust the weight parameter (e.g., `2.0f`) to control how much emphasis is placed on flip resets relative to other rewards.

2. **Internal Parameters**: If needed, you can modify the internal parameters in the `FlipResetReward.cpp` file:
   - `fromWallRatio`: Controls how much being away from walls is rewarded
   - `heightRatio`: Controls the optimal height
   - `bottomBallRatio`: Controls the importance of having the bottom of the car facing the ball
   - The `40.0f` factor in the final calculation can be adjusted to change the overall sensitivity

## Training Tips

1. **Combine with Other Rewards**: The flip reset reward works best when combined with basic rewards like ball touch and goal rewards.

2. **Curriculum Learning**: Consider starting with a lower weight for the flip reset reward and gradually increasing it as training progresses.

3. **Observation Space**: Make sure your observation space includes the car's orientation and the relative position of the ball.

4. **Patience**: Flip resets are complex mechanics that may take a long time to learn. Be patient with your agent's training.

## Example

See `FlipResetReward_Example.cpp` for a complete example of how to use this reward.