#pragma once

#include "richard/types.hpp"
#include <cstdint>
#include <string>
#include <sstream>
#include <set>
#include <algorithm>

namespace richard {

#define STR(x) [&]() {\
  std::stringstream ss; \
  ss << x; \
  return ss.str(); \
}()

using hashedString_t = size_t;

hashedString_t hashString(const std::string& value);

constexpr size_t calcProduct(const Size3& s) {
  return s[0] * s[1] * s[2];
}

constexpr size_t calcSum(const Size3& s) {
  return s[0] + s[1] + s[2];
}

template<class T>
void setDifference(std::set<T>& A, const std::set<T>& B) {
  for (auto i : B) {
    A.erase(i);
  }
}

template<class T>
void setDifference(const std::set<T>& A, const std::set<T>& B, std::set<T>& result) {
  std::set_difference(A.begin(), A.end(), B.begin(), B.end(), std::inserter(result, result.end()));
}

template<class T>
void setUnion(std::set<T>& A, const std::set<T>& B) {
  A.insert(B.begin(), B.end());
}

template<class T>
void setUnion(const std::set<T>& A, const std::set<T>& B, std::set<T>& result) {
  std::set_union(A.begin(), A.end(), B.begin(), B.end(), std::inserter(result, result.end()));
}

template<class T>
void setIntersection(std::set<T>& A, const std::set<T>& B) {
  std::set_intersection(A.begin(), A.end(), B.begin(), B.end());
}

template<class T>
void setIntersection(const std::set<T>& A, const std::set<T>& B, std::set<T>& result) {
  std::set_intersection(A.begin(), A.end(), B.begin(), B.end(),
    std::inserter(result, result.end()));
}

std::ostream& operator<<(std::ostream& os, const Size3& size);

uint32_t majorVersion();
uint32_t minorVersion();
std::string versionString();

}
