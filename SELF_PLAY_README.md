# Self-Play with Policy Pool

This feature allows your agent to play against older versions of itself during training. This is a form of curriculum learning that can help your agent develop more robust strategies.

## How It Works

1. The system maintains a pool of older policies (previous versions of your agent).
2. During training, with a configurable probability (default 20%), your agent will play against one of these older policies instead of against itself.
3. The pool is updated at regular intervals, adding the current policy to the pool.
4. If the pool exceeds the maximum size, the oldest policy is removed.

## Configuration

To enable and configure self-play, modify the following settings in your `LearnerConfig`:

```cpp
// Policy pool settings for self-play
bool enablePolicyPool = true; // Enable self-play with policy pool
int policyPoolSize = 5; // Maximum number of old policies to keep in the pool
float selfPlayRatio = 0.2f; // Ratio of games against older policies (0.0-1.0)
int64_t timestepsPerPoolUpdate = 500 * 1000; // How often to add a new policy to the pool
```

## Implementation Details

The implementation consists of several components:

1. **PolicyPool**: A class that manages a collection of older policies.
2. **ThreadAgentManager**: Modified to support using policies from the pool.
3. **Learner**: Updated to initialize and update the policy pool.

## How to Use

1. Set `enablePolicyPool = true` in your `LearnerConfig`.
2. Adjust the other parameters as needed:
   - `policyPoolSize`: How many old policies to keep (more = more variety but more memory usage)
   - `selfPlayRatio`: How often to play against old policies (0.2 = 20% of games)
   - `timestepsPerPoolUpdate`: How often to add a new policy to the pool

## Benefits

- **Prevents Forgetting**: Your agent maintains performance against strategies it has seen before.
- **Avoids Local Optima**: Helps prevent your agent from overfitting to the current version of itself.
- **Curriculum Learning**: Naturally creates a curriculum where your agent faces progressively stronger opponents.

## Limitations

- **Memory Usage**: Each policy in the pool requires additional memory.
- **Computation**: Using policies from the pool adds some computational overhead.
- **Simple Implementation**: The current implementation assumes standard team setups (1v1, 2v2, 3v3) with equal team sizes.

## Future Improvements

- Support for asymmetric team sizes
- More sophisticated opponent selection strategies
- Tracking performance against specific older versions