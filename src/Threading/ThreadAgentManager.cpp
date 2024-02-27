#include "ThreadAgentManager.h"

void RLGPC::ThreadAgentManager::CreateAgents(EnvCreateFn func, int amount, int gamesPerAgent) {
	for (int i = 0; i < amount; i++) {
		auto agent = new ThreadAgent(this, gamesPerAgent, func);
		agents.push_back(agent);
	}
}

RLGPC::GameTrajectory RLGPC::ThreadAgentManager::CollectTimesteps(uint64_t amount) {

	// We will just wait in this loop until our agents have collected enough total timesteps
	while (true) {
		uint64_t totalSteps = 0;
		for (auto agent : agents)
			for (auto& trajSet : agent->trajectories)
				for (auto& traj : trajSet)
					totalSteps += traj.size;

		if (totalSteps >= amount)
			break;

		// "waiter! waiter! more timesteps please!"

		// TODO: Possibly sub-optimal waiting solution:
		// This seems ok, but timestep collection only happens every once in a while (timings are very variable to config), 
		//	so maybe its actually smarter to run sleep(1) or something?
		// 
		// Could also just have the agents keep track of total step collection and unlock this thread?
		std::this_thread::yield();
	}

	// Our agents have collected the timesteps we need
	 
	// Combine all of their trajectories into one long trajectory
	// We will return this giant trajectory to the learner, for both computing GAE and 
	GameTrajectory result = {};
	for (auto agent : agents) {
		agent->trajMutex.lock();
		for (auto& trajSet : agent->trajectories) {
			for (auto& traj : trajSet) {
				if (traj.size > 0) {
					// If the last timestep is not a done, mark it as truncated
					// The GAE needs to know when the environment state stops being continuous
					// This happens either because the environment reset (i.e. goal scored), called "done",
					//	or the data got cut short, called "truncated"
					traj.data.truncateds[traj.size - 1] = (traj.data.dones[traj.size - 1].item<float>() == 0);

					result.Append(traj);
					traj.Clear();
				} else {
					// Kinda lame but does happen
				}
			}
		}
		agent->trajMutex.unlock();
	}

	return result;
}

void RLGPC::ThreadAgentManager::GetMetrics(Report& report) {
	AvgTracker avgStepRew, avgEpRew;
	for (auto agent : agents) {
		for (auto game : agent->gameInsts) {
			avgStepRew += game->avgStepRew.Get();
			avgEpRew += game->avgEpRew.Get();
		}
	}
	
	report["Average Step Reward"] = avgStepRew.Get();
	report["Average Episode Reward"] = avgEpRew.Get();

	ThreadAgent::Times avgTimes = {};

	for (ThreadAgent* agent : agents)
		for (auto itr1 = avgTimes.begin(), itr2 = agent->times.begin(); itr1 != avgTimes.end(); itr1++, itr2++)
			*itr1 += *itr2;
	
	for (double& time : avgTimes)
		time /= agents.size();

	report["Env Step Time"] = avgTimes.envStepTime;
	// NOTE: Because of non-blocking mode, a good portion of policy inference time is waited when appending trajectories
	//	This means the trajectory append time is not correct at all, so this is a temporary solution
	report["Policy Infer Time"] = avgTimes.policyInferTime + avgTimes.trajAppendTime;
}

void RLGPC::ThreadAgentManager::ResetMetrics() {
	for (auto agent : agents) {
		agent->times = {};
		for (auto game : agent->gameInsts)
			game->ResetAvgs();
	}
}