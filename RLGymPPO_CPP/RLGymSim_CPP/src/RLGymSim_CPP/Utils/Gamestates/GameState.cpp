#include "GameState.h"

#include "../../Math.h"

using namespace RLGSC;

static int boostPadIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
static bool boostPadIndexMapBuilt = false;
static std::mutex boostPadIndexMapMutex = {};
void _BuildBoostPadIndexMap(Arena* arena) {
	constexpr const char* ERROR_PREFIX = "_BuildBoostPadIndexMap(): ";
	RG_LOG("Building boost pad index map...");

	if (arena->_boostPads.size() != CommonValues::BOOST_LOCATIONS_AMOUNT) {
		RG_ERR_CLOSE(
			ERROR_PREFIX << "Arena boost pad count does not match CommonValues::BOOST_LOCATIONS_AMOUNT " <<
			"(" << arena->_boostPads.size() << "/" << CommonValues::BOOST_LOCATIONS_AMOUNT << ")"
		);
	}
	
	bool found[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		Vec targetPos = CommonValues::BOOST_LOCATIONS[i];
		for (int j = 0; j < arena->_boostPads.size(); j++) {
			Vec padPos = arena->_boostPads[j]->pos;

			if (padPos.DistSq2D(targetPos) < 10) {
				if (!found[i]) {
					found[i] = true;
					boostPadIndexMap[i] = j;
				} else {
					RG_ERR_CLOSE(
						ERROR_PREFIX << "Matched duplicate boost pad at " << targetPos << "=" << padPos
					);
				}
				break;
			}
		}

		if (!found[i])
			RS_ERR_CLOSE(ERROR_PREFIX << "Failed to find matching pad at " << targetPos);
	}

	RG_LOG(" > Done");
	boostPadIndexMapBuilt = true;
}

void RLGSC::GameState::UpdateFromArena(Arena* arena) {

	ballState = arena->ball->GetState();
	ball = PhysObj(ballState);
	ballInv = ball.Invert();

	players.resize(arena->_cars.size());

	auto carItr = arena->_cars.begin();
	for (int i = 0; i < players.size(); i++) {
		auto& player = players[i];
		player.UpdateFromCar(*carItr, arena->tickCount);
		if (player.ballTouched)
			lastTouchCarID = player.carId;

		carItr++;
	}

	if (!boostPadIndexMapBuilt) {
		boostPadIndexMapMutex.lock();
		// Check again? This seems stupid but also makes sense to me
		//	Without this, we could lock as the index map is building, then go build again
		//	I would like to keep the mutex inside the if statement so it is only checked a few times
		if (!boostPadIndexMapBuilt) 
			_BuildBoostPadIndexMap(arena);
		boostPadIndexMapMutex.unlock();
	}

	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		boostPads[i] = arena->_boostPads[boostPadIndexMap[i]]->GetState().isActive;
		boostPadsInv[i] = arena->_boostPads[boostPadIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1]]->GetState().isActive;
	}

	// Update goal scoring
	// If you don't have a GoalScoreCondition then that's not my problem lmao
	if (Math::IsBallScored(ball.pos))
		scoreLine[(int)RS_TEAM_FROM_Y(ball.pos.y)]++;
}