#pragma once
#include <RLGymPPO_CPP/Framework.h>

namespace RLGPC {
	// Modified version of https://stackoverflow.com/questions/26516683/reusing-thread-in-loop-c
	struct ThreadPool {

		std::mutex lockMutex = {};
		std::condition_variable condVar = {};
		bool shouldShutdown = false;
		std::queue<std::function<void(void)>> _jobs = {};
		std::vector<std::thread> threads = {};
		int _activeJobCounter = 0;

		ThreadPool(int numThreads) {
			// Create the specified number of threads
			threads.reserve(numThreads);
			for (int i = 0; i < numThreads; ++i)
				threads.emplace_back(std::bind(&ThreadPool::_ThreadEntry, this, i));
		}

		~ThreadPool() {
			{
				// Unblock any threads and tell them to stop
				std::unique_lock <std::mutex> l(lockMutex);

				shouldShutdown = true;
				condVar.notify_all();
			}

			// Wait for all threads to stop
			for (auto& thread : threads)
				thread.join();
		}

		void StartJob(std::function <void(void)> func) {
			// Place a job on the queue and unblock a thread
			std::unique_lock<std::mutex> lock(lockMutex);

			_activeJobCounter++;

			_jobs.emplace(std::move(func));
			condVar.notify_one();
		}

		int GetNumRunningJobs() {
			std::unique_lock<std::mutex> lock(lockMutex);
			return _activeJobCounter;
		}

		void _ThreadEntry(int i) {
			std::function<void(void)> jobFunc;

			while (true) {
				{
					std::unique_lock<std::mutex> lock(lockMutex);

					while (!shouldShutdown && _jobs.empty())
						condVar.wait(lock);

					if (_jobs.empty()) {
						// Shutdown is true, no jobs left
						return;
					}

					jobFunc = std::move(_jobs.front());
					_jobs.pop();
				}

				// Do the job without holding any locks
				jobFunc();

				{
					std::unique_lock<std::mutex> lock(lockMutex);
					_activeJobCounter--;
				}
			}
		}
	};
}