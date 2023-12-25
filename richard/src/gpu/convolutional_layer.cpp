#include "gpu/convolutional_layer.hpp"
#include "util.hpp"

namespace richard {
namespace gpu {

ConvolutionalLayer::ConvolutionalLayer(Gpu& gpu, const nlohmann::json& obj, size_t inputW,
  size_t inputH, size_t inputDepth) {

  // TODO
}

ConvolutionalLayer::ConvolutionalLayer(Gpu& gpu, const nlohmann::json& obj, std::istream& stream,
  size_t inputW, size_t inputH, size_t inputDepth) {

  // TODO
}

void ConvolutionalLayer::allocateGpuResources(GpuBufferHandle inputBuffer,
  GpuBufferHandle statusBuffer, const Layer* nextLayer, GpuBufferHandle) {

  // TODO
}

size_t ConvolutionalLayer::size() const {
  // TODO
}

Triple ConvolutionalLayer::outputSize() const {
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
