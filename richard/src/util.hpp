#pragma once

#include <cassert>
#include <map>
#include <string>

#define ASSERT(X) assert(X)

std::map<std::string, std::string> readKeyValuePairs(std::istream& s);
