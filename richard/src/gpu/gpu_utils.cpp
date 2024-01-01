#include "gpu/gpu_utils.hpp"
#include "exception.hpp"

namespace richard {
namespace gpu {

void optimumWorkgroups(const Size3& workSize, Size3& workgroupSize, Size3& numWorkgroups) {
  const size_t maxWorkgroupSize = 64;

  workgroupSize = {
    static_cast<uint32_t>(std::min(workSize[0], maxWorkgroupSize)),
    static_cast<uint32_t>(std::min(workSize[1], maxWorkgroupSize)),
    workSize[2]
  };

  numWorkgroups = {
    workSize[0] / workgroupSize[0],
    workSize[1] / workgroupSize[1],
    1
  };

  ASSERT_MSG(workgroupSize[0] * numWorkgroups[0] == workSize[0],
    "Work size " << workSize[0] << " is not divisible by workgroup size " << workgroupSize[0]);

  ASSERT_MSG(workgroupSize[1] * numWorkgroups[1] == workSize[1],
    "Work size " << workSize[1] << " is not divisible by workgroup size " << workgroupSize[1]);
}

}
}
