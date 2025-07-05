#pragma once
#include "../FrameworkTorch.h"
#include "../PPO/DiscretePolicy.h"
#include <vector>
#include <random>
#include <filesystem>

namespace RLGPC {
    // Class to manage a pool of older policies for self-play
    class PolicyPool {
    public:
        struct PooledPolicy {
            DiscretePolicy* policy;
            int64_t timesteps;
            
            PooledPolicy(DiscretePolicy* policy, int64_t timesteps) : 
                policy(policy), timesteps(timesteps) {}
        };
        
        std::vector<PooledPolicy> policies;
        int maxPolicies;
        float selfPlayRatio; // Ratio of games against older policies (0.0-1.0)
        std::mt19937 rng;
        
        PolicyPool(int maxPolicies = 5, float selfPlayRatio = 0.2f, int seed = 123) : 
            maxPolicies(maxPolicies), selfPlayRatio(selfPlayRatio) {
            rng.seed(seed);
        }
        
        // Add a policy to the pool
        void AddPolicy(DiscretePolicy* policy, int64_t timesteps) {
            // Create a deep copy of the policy
            DiscretePolicy* policyCopy = new DiscretePolicy(*policy);
            policies.emplace_back(policyCopy, timesteps);
            
            // If we have too many policies, remove the oldest one
            if (policies.size() > maxPolicies) {
                delete policies.front().policy;
                policies.erase(policies.begin());
            }
        }
        
        // Get a random policy from the pool
        DiscretePolicy* GetRandomPolicy() {
            if (policies.empty()) {
                return nullptr;
            }
            
            std::uniform_int_distribution<int> dist(0, policies.size() - 1);
            return policies[dist(rng)].policy;
        }
        
        // Decide whether to use a policy from the pool based on the self-play ratio
        bool ShouldUsePooledPolicy() {
            if (policies.empty()) {
                return false;
            }
            
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            return dist(rng) < selfPlayRatio;
        }
        
        // Clean up
        ~PolicyPool() {
            for (auto& policy : policies) {
                delete policy.policy;
            }
        }
    };
}