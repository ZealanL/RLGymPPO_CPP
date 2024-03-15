#pragma once

#define RS_DONT_LOG // Prevent annoying log spam
#include "../RocketSim/src/RocketSim.h"
#include "../RocketSim/src/Sim/GameEventTracker/GameEventTracker.h"

// Define our own log
#define RG_LOG(s) { std::cout << s << std::endl; }

#define RG_NO_COPY(className) \
className(const className&) = delete;  \
className& operator= (const className&) = delete

#define RG_ERR_CLOSE(s) { std::string _errorStr = RS_STR("RG FATAL ERROR: " << s); RG_LOG(_errorStr); throw std::runtime_error(_errorStr); exit(EXIT_FAILURE); }