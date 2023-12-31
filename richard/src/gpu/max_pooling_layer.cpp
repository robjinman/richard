#include "gpu/max_pooling_layer.hpp"
#include "gpu/gpu_utils.hpp"
#include "utils.hpp"

namespace richard {
namespace gpu {

MaxPoolingLayer::MaxPoolingLayer(Gpu& gpu, const nlohmann::json& obj, const Size3& inputShape)
  : m_gpu(gpu)
  , m_inputW(inputShape[0])
  , m_inputH(inputShape[1])
  , m_inputDepth(inputShape[2]) {

  std::array<size_t, 2> regionSize = getOrThrow(obj, "regionSize").get<std::array<size_t, 2>>();
  m_regionW = regionSize[0];
  m_regionH = regionSize[1];

  ASSERT_MSG(m_inputW % m_regionW == 0,
    "Region width " << m_regionW << " does not divide input width " << m_inputW);
  ASSERT_MSG(m_inputH % m_regionH == 0,
    "Region height " << m_regionH << " does not divide input height " << m_inputH);
}

void MaxPoolingLayer::allocateGpuBuffers() {
  size_t inputSize = m_inputW * m_inputH * m_inputDepth;

  m_bufferZ = m_gpu.allocateBuffer(size() * sizeof(netfloat_t), GpuBufferFlags::large);
  m_bufferInputDelta = m_gpu.allocateBuffer(inputSize * sizeof(netfloat_t), GpuBufferFlags::large);
  m_bufferMask = m_gpu.allocateBuffer(inputSize * sizeof(netfloat_t), GpuBufferFlags::large);
}

void MaxPoolingLayer::createGpuShaders(GpuBufferHandle inputBuffer, GpuBufferHandle,
  const Layer* nextLayer, GpuBufferHandle) {

  DBG_ASSERT(nextLayer != nullptr);

  GpuBufferBindings evalForwardBuffers{
    inputBuffer,
    m_bufferZ.handle
  };

  GpuBufferBindings trainForwardBuffers{
    inputBuffer,
    m_bufferZ.handle,
    m_bufferMask.handle
  };

  GpuBufferBindings backpropBuffers{
    m_bufferZ.handle,
    m_bufferMask.handle,
    nextLayer->deltaBuffer()
  };

  Size3 workgroupSize;
  Size3 numWorkgroups;
  optimumWorkgroups(outputSize(), workgroupSize, numWorkgroups);

  SpecializationConstants evalForwardConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_regionW) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_regionH) }
  };

  SpecializationConstants trainForwardConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_regionW) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_regionH) }
  };

  SpecializationConstants backpropConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_regionW) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_regionH) }
  };

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string evalForwardSrc = loadFile("./shaders/max_pooling_eval_forward.glsl");
  const std::string trainForwardSrc = loadFile("./shaders/max_pooling_train_forward.glsl");
  const std::string backpropSrc = loadFile("./shaders/max_pooling_backprop.glsl");

  m_evalForwardShader = m_gpu.compileShader(evalForwardSrc, evalForwardBuffers,
    evalForwardConstants, workgroupSize, numWorkgroups, includesDir);
  m_trainForwardShader = m_gpu.compileShader(trainForwardSrc, trainForwardBuffers,
    trainForwardConstants, workgroupSize, numWorkgroups, includesDir);
  m_backpropShader = m_gpu.compileShader(backpropSrc, backpropBuffers, backpropConstants,
    workgroupSize, numWorkgroups, includesDir);
}

size_t MaxPoolingLayer::size() const {
  return calcProduct(outputSize());
}

Size3 MaxPoolingLayer::outputSize() const {
  return { m_inputW / m_regionW, m_inputH / m_regionH, m_inputDepth };
}

void MaxPoolingLayer::evalForward() {
  m_gpu.queueShader(m_evalForwardShader);
}

void MaxPoolingLayer::trainForward() {
  m_gpu.queueShader(m_trainForwardShader);
}

void MaxPoolingLayer::backprop() {
  m_gpu.queueShader(m_backpropShader);
}

void MaxPoolingLayer::updateParams() {}

GpuBufferHandle MaxPoolingLayer::outputBuffer() const {
  return m_bufferZ.handle;
}

GpuBufferHandle MaxPoolingLayer::weightsBuffer() const {
  EXCEPTION("Max pooling layer does not have a weights buffer");
}

GpuBufferHandle MaxPoolingLayer::deltaBuffer() const {
  EXCEPTION("Max pooling layer does not have a delta buffer");
}

GpuBufferHandle MaxPoolingLayer::inputDeltaBuffer() const {
  return m_bufferInputDelta.handle;
}

GpuBufferHandle MaxPoolingLayer::maskBuffer() const {
  return m_bufferMask.handle;
}

void MaxPoolingLayer::retrieveBuffers() {}

void MaxPoolingLayer::writeToStream(std::ostream&) const {}

}
}
