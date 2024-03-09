#pragma once
#include "../../Framework.h"

namespace RLGSC {
	typedef std::vector<float> FList;
	typedef std::vector<std::vector<float>> FList2;
	typedef std::vector<int> IList;
	typedef std::vector<std::vector<int>> IList2;
}

// FList operators
inline RLGSC::FList& operator +=(RLGSC::FList& list, float val) {
	list.push_back(val);
	return list;
}

inline RLGSC::FList& operator +=(RLGSC::FList& list, const Vec& val) {
	list.push_back(val.x);
	list.push_back(val.y);
	list.push_back(val.z);
	return list;
}

inline RLGSC::FList& operator +=(RLGSC::FList& list, const std::initializer_list<float>& other) {
	list.insert(list.end(), other.begin(), other.end());
	return list;
}

inline RLGSC::FList& operator +=(RLGSC::FList& list, const RLGSC::FList& other) {
	list.insert(list.end(), other.begin(), other.end());
	return list;
}