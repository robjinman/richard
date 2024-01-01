#pragma once

#include "types.hpp"

namespace richard {
namespace gpu {

void optimumWorkgroups(const Size3& workSize, Size3& workgroupSize, Size3& numWorkgroups);

}
}
