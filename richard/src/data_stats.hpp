#pragma once

#include "math.hpp"

struct DataStats {
  DataStats(const Vector& min, const Vector& max)
    : min(min)
    , max(max) {}

  Vector min; // Min/max values of every dimension
  Vector max;
};
