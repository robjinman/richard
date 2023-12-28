#include "gpu/max_pooling_layer.hpp"
#include "util.hpp"

namespace richard {
namespace gpu {

MaxPoolingLayer::MaxPoolingLayer(Gpu& gpu, const nlohmann::json& obj, size_t inputW, size_t inputH,
  size_t inputDepth) {

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

Triple MaxPoolingLayer::outputSize() const {
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
