#include "gpu/max_pooling_layer.hpp"
#include "utils.hpp"

namespace richard {
namespace gpu {

MaxPoolingLayer::MaxPoolingLayer(Gpu& gpu, const nlohmann::json& obj, const Size3& inputShape) {
  // TODO
}

void MaxPoolingLayer::allocateGpuBuffers() {
  // TODO
}

void MaxPoolingLayer::createGpuShaders(GpuBufferHandle inputBuffer,
  GpuBufferHandle statusBuffer, const Layer* nextLayer, GpuBufferHandle) {

  // TODO
}

size_t MaxPoolingLayer::size() const {
  // TODO
}

Size3 MaxPoolingLayer::outputSize() const {
  // TODO
}

void MaxPoolingLayer::evalForward() {
  // TODO
}

void MaxPoolingLayer::trainForward() {
  // TODO
}

void MaxPoolingLayer::backprop() {
  // TODO
}

void MaxPoolingLayer::updateParams() {
  // TODO
}

GpuBufferHandle MaxPoolingLayer::outputBuffer() const {
  // TODO
}

GpuBufferHandle MaxPoolingLayer::weightsBuffer() const {
  // TODO
}

GpuBufferHandle MaxPoolingLayer::deltaBuffer() const {
  // TODO
}

void MaxPoolingLayer::retrieveBuffers() {
  // TODO
}

void MaxPoolingLayer::writeToStream(std::ostream& stream) const {
  // TODO
}

}
}
