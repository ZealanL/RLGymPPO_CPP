#pragma once
#include "../Framework.h"

namespace RLGPC {
	struct Report {
		typedef double Val;
		std::map<std::string, Val> data;
		
		Report() = default;

		Val& operator[](const std::string& key) {
			return data[key];
		}

		Val operator[](const std::string& key) const {
			return data.at(key);
		}

		bool Has(const std::string& key) const {
			return data.find(key) != data.end();
		}

		void Accum(const std::string& key, Val val) {
			if (Has(key)) {
				data[key] += val;
			} else {
				data[key] = val;
			}
		}

		std::string SingleToString(const std::string& key, bool digitCommas = false) const {

			// https://stackoverflow.com/a/7277333
			class comma_numpunct : public std::numpunct<char>
			{
			protected:
				virtual char do_thousands_sep() const {
					return ',';
				}

				virtual std::string do_grouping() const {
					return "\03";
				}
			};
			static std::locale commaLocale(std::locale(), new comma_numpunct());

			std::stringstream stream;
			if (digitCommas)
				stream.imbue(commaLocale);

			stream << key << ": " << (*this)[key];
			return stream.str();
		}

		std::string ToString(bool digitCommas = false, const std::string& prefix = {}) const {
			std::stringstream stream;
			for (auto pair : data) {
				stream << prefix << SingleToString(pair.first, digitCommas) << std::endl;
			}
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