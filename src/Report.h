#pragma once
#include "Framework.h"

namespace RLGPC {
	struct Report {
		std::unordered_map<std::string, double> doubleData;
		std::unordered_map<std::string, int64_t> intData;

		void Add(std::string name, float data) {
			doubleData[name] = data;
		}

		void Add(std::string name, double data) {
			doubleData[name] = data;
		}

		void Add(std::string name, int64_t data) {
			intData[name] = data;
		}

		void Add(std::string name, int data) {
			intData[name] = data;
		}

		void Add(std::string name, bool data) {
			intData[name] = data;
		}

		std::string ToString(const std::string& prefix = {}) {
			std::stringstream stream;
			for (auto pair : doubleData)
				stream << prefix << pair.first << ": " << pair.second << std::endl;
			for (auto pair : intData)
				stream << prefix << pair.first << ": " << RG_COMMA_INT(pair.second) << std::endl;
			return stream.str();
		}
	};
}