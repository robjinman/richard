#include "gpu/convolutional_layer.hpp"
#include "util.hpp"

namespace richard {
namespace gpu {

ConvolutionalLayer::ConvolutionalLayer(Gpu& gpu, const nlohmann::json& obj,
  const Size3& inputShape) {

  // TODO
}

ConvolutionalLayer::ConvolutionalLayer(Gpu& gpu, const nlohmann::json& obj, std::istream& stream,
  const Size3& inputShape) {

  // TODO
}

void ConvolutionalLayer::allocateGpuBuffers() {
  // TODO
}

void ConvolutionalLayer::createGpuShaders(GpuBufferHandle inputBuffer,
  GpuBufferHandle statusBuffer, const Layer* nextLayer, GpuBufferHandle) {

  // TODO
}

size_t ConvolutionalLayer::size() const {
  // TODO
}

Size3 ConvolutionalLayer::outputSize() const {
  // TODO
}

void ConvolutionalLayer::evalForward() {
  // TODO
}

void ConvolutionalLayer::trainForward() {
  // TODO
}

void ConvolutionalLayer::backprop() {
  // TODO
}

void ConvolutionalLayer::updateParams() {
  // TODO
}

GpuBufferHandle ConvolutionalLayer::outputBuffer() const {
  // TODO
}

GpuBufferHandle ConvolutionalLayer::weightsBuffer() const {
  // TODO
}

GpuBufferHandle ConvolutionalLayer::deltaBuffer() const {
  // TODO
}

void ConvolutionalLayer::retrieveBuffers() {
  // TODO
}

void ConvolutionalLayer::writeToStream(std::ostream& stream) const {
  // TODO
}

}
}
