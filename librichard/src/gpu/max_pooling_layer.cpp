#include "richard/gpu/max_pooling_layer.hpp"
#include "richard/utils.hpp"
#include "richard/file_system.hpp"
#include "richard/platform_paths.hpp"
#include "richard/config.hpp"

namespace richard {
namespace gpu {

MaxPoolingLayer::MaxPoolingLayer(Gpu& gpu, FileSystem& fileSystem,
  const PlatformPaths& platformPaths, const Config& config, const Size3& inputShape)
  : m_gpu(gpu)
  , m_fileSystem(fileSystem)
  , m_platformPaths(platformPaths)
  , m_inputW(inputShape[0])
  , m_inputH(inputShape[1])
  , m_inputDepth(inputShape[2]) {

  auto regionSize = config.getNumberArray<size_t, 2>("regionSize");
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

  createEvalForwardShader(inputBuffer);
  createTrainForwardShader(inputBuffer);
  createBackpropShader(nextLayer);
}

void MaxPoolingLayer::createEvalForwardShader(GpuBufferHandle inputBuffer) {
  GpuBufferBindings buffers{
    { inputBuffer, BufferAccessMode::read },
    { m_bufferZ.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_regionW) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_regionH) }
  };

  std::string shaderName = "max_pooling_eval_forward.spv";
  auto shaderCode = m_fileSystem.loadBinaryFile(m_platformPaths.get("shaders", shaderName));

  Size3 workSize = outputSize();

  m_evalForwardShader = m_gpu.addShader(shaderName, shaderCode, buffers, constants, workSize);
}

void MaxPoolingLayer::createTrainForwardShader(GpuBufferHandle inputBuffer) {
  GpuBufferBindings buffers{
    { inputBuffer, BufferAccessMode::read },
    { m_bufferZ.handle, BufferAccessMode::write },
    { m_bufferMask.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_regionW) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_regionH) }
  };

  std::string shaderName = "max_pooling_train_forward.spv";
  auto shaderCode = m_fileSystem.loadBinaryFile(m_platformPaths.get("shaders", shaderName));

  Size3 workSize = outputSize();

  m_trainForwardShader = m_gpu.addShader(shaderName, shaderCode, buffers, constants, workSize);
}

void MaxPoolingLayer::createBackpropShader(const Layer* nextLayer) {
  GpuBufferBindings buffers{
    { nextLayer->inputDeltaBuffer(), BufferAccessMode::read },
    { m_bufferMask.handle, BufferAccessMode::read },
    { m_bufferInputDelta.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_regionW) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_regionH) }
  };

  std::string shaderName = "max_pooling_backprop.spv";
  auto shaderCode = m_fileSystem.loadBinaryFile(m_platformPaths.get("shaders", shaderName));

  Size3 workSize = outputSize();

  m_backpropShader = m_gpu.addShader(shaderName, shaderCode, buffers, constants, workSize);
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

GpuBufferHandle MaxPoolingLayer::test_maskBuffer() const {
  return m_bufferMask.handle;
}

void MaxPoolingLayer::retrieveBuffers() {}

void MaxPoolingLayer::writeToStream(std::ostream&) const {}

}
}
