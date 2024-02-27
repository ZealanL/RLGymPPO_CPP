#pragma once
#include "../Framework.h"

namespace RLGPC {
	struct Report {
		struct Val {
			bool isDouble;
			union {
				double d;
				int64_t i;
			};

			Val() {
				isDouble = true;
				d = 0;
			}

			template<
				typename T,
				std::enable_if_t<std::is_floating_point<T>::value, bool> = true
			>
			Val(T val) {
				isDouble = true;
				d = (double)val;
			}

			template<
				typename T, 
				std::enable_if_t<std::is_integral<T>::value, bool> = true
			>
			Val(T val) {
				isDouble = false;
				i = (int64_t)val;
			}

			std::string ToString() const {
				if (isDouble) {
					return std::to_string(d);
				} else {
					return RG_COMMA_INT(i);
				}
			}
		};
		std::map<std::string, Val> data;
		
		Report() = default;

		Val& operator[](const std::string& key) {
			return data[key];
		}

		Val operator[](const std::string& key) const {
			return data.at(key);
		}

		std::string ToString(const std::string& prefix = {}) const {
			std::stringstream stream;
			for (auto pair : data)
				stream << prefix << pair.first << ": " << pair.second.ToString() << std::endl;
			return stream.str();
		}

		Report operator+(const Report& other) const {
			Report newReport = *this;
			newReport.data.insert(other.data.begin(), other.data.end());
			return newReport;
		}

		Report& operator+=(const Report& other) {
			*this = *this + other;
			return *this;
		}
	};
}