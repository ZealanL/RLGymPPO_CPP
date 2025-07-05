# Integration Guide for Self-Play Feature

This guide explains how to integrate the self-play with policy pool feature into the RLGymPPO_CPP codebase.

## Files Created

1. `/workspace/RLGymPPO_CPP/RLGymPPO_CPP/src/private/RLGymPPO_CPP/Util/PolicyPool.h`
2. `/workspace/RLGymPPO_CPP/RLGymPPO_CPP/src/private/RLGymPPO_CPP/Threading/ThreadAgent_modified.cpp`
3. `/workspace/RLGymPPO_CPP/RLGymPPO_CPP/src/public/RLGymPPO_CPP/Learner_modified.cpp`

## Files Modified

1. `/workspace/RLGymPPO_CPP/RLGymPPO_CPP/src/public/RLGymPPO_CPP/LearnerConfig.h`
2. `/workspace/RLGymPPO_CPP/RLGymPPO_CPP/src/public/RLGymPPO_CPP/Learner.h`
3. `/workspace/RLGymPPO_CPP/RLGymPPO_CPP/src/private/RLGymPPO_CPP/Threading/ThreadAgentManager.h`

## Integration Steps

### 1. Add the PolicyPool class

Copy the `PolicyPool.h` file to the specified location.

### 2. Update LearnerConfig.h

The changes to `LearnerConfig.h` have already been applied. They add the following configuration options:

```cpp
// Policy pool settings for self-play
bool enablePolicyPool = false; // Enable self-play with policy pool
int policyPoolSize = 5; // Maximum number of old policies to keep in the pool
float selfPlayRatio = 0.2f; // Ratio of games against older policies (0.0-1.0)
int64_t timestepsPerPoolUpdate = 500 * 1000; // How often to add a new policy to the pool
```

### 3. Update Learner.h

The changes to `Learner.h` have already been applied. They add:
- A `PolicyPool* policyPool` member variable
- A `timestepsSincePoolUpdate` counter
- A new `UpdatePolicyPool()` method

### 4. Update ThreadAgentManager.h

The changes to `ThreadAgentManager.h` have already been applied. They add:
- A `PolicyPool* policyPool` member variable
- A `SetPolicyPool(PolicyPool* pool)` method

### 5. Replace ThreadAgent.cpp with ThreadAgent_modified.cpp

The `ThreadAgent_modified.cpp` file contains the modified version of `ThreadAgent.cpp` that supports using policies from the pool. Replace the original file with this modified version:

```bash
mv /workspace/RLGymPPO_CPP/RLGymPPO_CPP/src/private/RLGymPPO_CPP/Threading/ThreadAgent_modified.cpp /workspace/RLGymPPO_CPP/RLGymPPO_CPP/src/private/RLGymPPO_CPP/Threading/ThreadAgent.cpp
```

### 6. Replace Learner.cpp with Learner_modified.cpp

The `Learner_modified.cpp` file contains the modified version of `Learner.cpp` that initializes and updates the policy pool. Replace the original file with this modified version:

```bash
mv /workspace/RLGymPPO_CPP/RLGymPPO_CPP/src/public/RLGymPPO_CPP/Learner_modified.cpp /workspace/RLGymPPO_CPP/RLGymPPO_CPP/src/public/RLGymPPO_CPP/Learner.cpp
```

## Testing the Integration

After integrating the changes, you can test the self-play feature by:

1. Creating a `LearnerConfig` with `enablePolicyPool = true`
2. Setting appropriate values for `policyPoolSize`, `selfPlayRatio`, and `timestepsPerPoolUpdate`
3. Running your training as usual

Example configuration:

```cpp
LearnerConfig config;
config.enablePolicyPool = true;
config.policyPoolSize = 5;
config.selfPlayRatio = 0.2f;
config.timestepsPerPoolUpdate = 500 * 1000;
```

## Troubleshooting

If you encounter issues after integration:

1. Check that all files have been properly replaced/updated
2. Verify that the `PolicyPool` class is being properly initialized in `Learner.cpp`
3. Make sure the `ThreadAgent.cpp` file contains the modified code that supports using policies from the pool
4. Check for any compilation errors related to the new code

## Notes

- The self-play feature is disabled by default (`enablePolicyPool = false`)
- The feature requires additional memory to store the old policies
- The current implementation assumes standard team setups (1v1, 2v2, 3v3) with equal team sizes