#include "gpu/gpu_utils.hpp"
#include "utils.hpp"
#include "exception.hpp"
//#include <iostream> // TODO

namespace richard {
namespace gpu {
namespace {

size_t maxValue(const Size3& size, size_t& value) {
  size_t index = 0;
  value = std::numeric_limits<size_t>::lowest();
  for (size_t i = 0; i < 3; ++i) {
    if (size[i] > value) {
      index = i;
      value = size[i];
    }
  }
  return index;
}

size_t lowestDivisor(size_t value) {
  for (size_t i = 2; i < value; ++i) {
    if (value % i == 0) {
      return i;
    }
  }
  return value;
}

//std::ostream& operator<<(std::ostream& os, const Size3& size) {
//  os << size[0] << ", " << size[1] << ", " << size[2];
//  return os;
//}

}

void optimumWorkgroups(const Size3& workSize, Size3& workgroupSize, Size3& numWorkgroups) {
  const size_t maxWorkgroupSize = 1536; // TODO: Query from Gpu class

  workgroupSize = workSize;
  numWorkgroups = { 1, 1, 1 };

  //std::cout << "Calculating optimum workgroups\n";
  //std::cout << "Work size: " << workSize << "\n";
  //std::cout << "Total: " << calcProduct(workSize) << "\n";

  while (calcProduct(workgroupSize) > maxWorkgroupSize) {
    size_t largest = 0;
    size_t i = maxValue(workgroupSize, largest);

    size_t scale = lowestDivisor(largest);

    workgroupSize[i] /= scale;
    numWorkgroups[i] *= scale;
  }

  //std::cout << "Workgroup size: " << workgroupSize << "\n";
  //std::cout << "Total: " << calcProduct(workgroupSize) << "\n";
  //std::cout << "Workgroups: " << numWorkgroups << "\n\n";

  ASSERT_MSG(workgroupSize[0] * numWorkgroups[0] == workSize[0],
    "Work size " << workSize[0] << " is not divisible by workgroup size " << workgroupSize[0]);

  ASSERT_MSG(workgroupSize[1] * numWorkgroups[1] == workSize[1],
    "Work size " << workSize[1] << " is not divisible by workgroup size " << workgroupSize[1]);
}

}
}
