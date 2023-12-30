#include "gpu/convolutional_layer.hpp"
#include "util.hpp"
#include "math.hpp"

namespace richard {
namespace gpu {

ConvolutionalLayer::ConvolutionalLayer(Gpu& gpu, const nlohmann::json& obj,
  const Size3& inputShape, bool isFirstLayer)
  : m_gpu(gpu) {

  initialize(obj, inputShape, isFirstLayer);
}

ConvolutionalLayer::ConvolutionalLayer(Gpu& gpu, const nlohmann::json& obj, std::istream& stream,
  const Size3& inputShape, bool isFirstLayer)
  : m_gpu(gpu) {

  initialize(obj, inputShape, isFirstLayer);

  size_t kernelSize = m_kernelSize[0] * m_kernelSize[1] * m_inputDepth;

  for (size_t i = 0; i < m_depth; ++i) {
    stream.read(reinterpret_cast<char*>(m_biasData.data() + i), sizeof(netfloat_t));
    stream.read(reinterpret_cast<char*>(m_kernelData.data() + i * kernelSize),
      kernelSize * sizeof(netfloat_t));
  }
}

void ConvolutionalLayer::initialize(const nlohmann::json& obj, const Size3& inputShape,
  bool isFirstLayer) {

  m_inputW = inputShape[0];
  m_inputH = inputShape[1];
  m_inputDepth = inputShape[2];
  m_kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();
  m_depth = getOrThrow(obj, "depth").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<netfloat_t>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<netfloat_t>();
  m_dropoutRate = getOrThrow(obj, "dropoutRate").get<netfloat_t>();
  m_isFirstLayer = isFirstLayer;
  m_kernelData = Vector(m_kernelSize[0] * m_kernelSize[1] * m_inputDepth * m_depth);
  m_biasData = Vector(m_depth);
}

void ConvolutionalLayer::allocateGpuBuffers() {
  GpuBufferFlags paramBuffersFlags = GpuBufferFlags::large
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;

  size_t kernelSize = m_kernelSize[0] * m_kernelSize[1] * m_inputDepth;
  size_t featureMapSizeBytes = calcProduct(outputSize()) * sizeof(netfloat_t);

  m_bufferK = m_gpu.allocateBuffer(m_depth * kernelSize * sizeof(netfloat_t), paramBuffersFlags);
  m_bufferB = m_gpu.allocateBuffer(m_depth * sizeof(netfloat_t), paramBuffersFlags);
  m_bufferZ = m_gpu.allocateBuffer(featureMapSizeBytes, GpuBufferFlags::large);
  m_bufferA = m_gpu.allocateBuffer(featureMapSizeBytes, GpuBufferFlags::large);
  m_bufferD = m_gpu.allocateBuffer(featureMapSizeBytes, GpuBufferFlags::large);
  m_bufferDeltaK = m_gpu.allocateBuffer(m_depth * kernelSize * sizeof(netfloat_t),
    GpuBufferFlags::large | GpuBufferFlags::hostWriteAccess);
  m_bufferDeltaB = m_gpu.allocateBuffer(m_depth * sizeof(netfloat_t),
    GpuBufferFlags::large | GpuBufferFlags::hostWriteAccess);

  m_kernelData.randomize(0.1);
  m_gpu.submitBufferData(m_bufferK.handle, m_kernelData.data());

  Vector deltaKData(m_kernelData.size());
  m_gpu.submitBufferData(m_bufferDeltaK.handle, deltaKData.data());

  m_gpu.submitBufferData(m_bufferB.handle, m_biasData.data());
  m_gpu.submitBufferData(m_bufferDeltaB.handle, m_biasData.data());
}

void ConvolutionalLayer::createGpuShaders(GpuBufferHandle inputBuffer, GpuBufferHandle statusBuffer,
  const Layer* nextLayer, GpuBufferHandle) {

  DBG_ASSERT(nextLayer != nullptr);

  GpuBufferBindings evalForwardBuffers{
    inputBuffer,
    m_bufferK.handle,
    m_bufferB.handle,
    m_bufferA.handle
  };

  GpuBufferBindings trainForwardBuffers{
    statusBuffer,
    inputBuffer,
    m_bufferK.handle,
    m_bufferB.handle,
    m_bufferZ.handle,
    m_bufferA.handle
  };

  GpuBufferBindings backpropBuffers{
    statusBuffer,
    inputBuffer,
    m_bufferK.handle,
    m_bufferB.handle,
    m_bufferZ.handle,
    m_bufferA.handle,
    m_bufferD.handle,
    nextLayer->weightsBuffer(),
    nextLayer->deltaBuffer(),
    m_bufferDeltaK.handle
  };

  GpuBufferBindings updateParamsBuffers{
    statusBuffer,
    m_bufferK.handle,
    m_bufferB.handle,
    m_bufferDeltaK.handle,
    m_bufferDeltaB.handle
  };

  const size_t maxWorkgroupSize = 64;
  const Size3& featureMapSize = outputSize();

  Size3 workgroupSize{
    static_cast<uint32_t>(std::min(featureMapSize[0], maxWorkgroupSize)),
    static_cast<uint32_t>(std::min(featureMapSize[1], maxWorkgroupSize)),
    m_depth
  };

  Size3 numWorkgroups{
    featureMapSize[0] / workgroupSize[0],
    featureMapSize[1] / workgroupSize[1],
    1
  };

  ASSERT_MSG(workgroupSize[0] * numWorkgroups[0] == featureMapSize[0],
    "Layer size " << featureMapSize[0] << " is not divisible by workgroup size "
    << workgroupSize[0]);

  ASSERT_MSG(workgroupSize[1] * numWorkgroups[1] == featureMapSize[1],
    "Layer size " << featureMapSize[1] << " is not divisible by workgroup size "
    << workgroupSize[1]);

  SpecializationConstants evalForwardConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_depth) }
  };

  SpecializationConstants trainForwardConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_depth) },
    { SpecializationConstant::Type::bool_type, m_isFirstLayer },
  //  { SpecializationConstant::Type::float_type, m_dropoutRate }
  };

  SpecializationConstants backpropConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_depth) },
    { SpecializationConstant::Type::bool_type, m_isFirstLayer },
  };

  SpecializationConstants updateParamsConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_depth) },
    { SpecializationConstant::Type::float_type, m_learnRate },
    { SpecializationConstant::Type::float_type, m_learnRateDecay },
  };

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string evalForwardSrc = loadFile("./shaders/convolutional_eval_forward.glsl");
  const std::string trainForwardSrc = loadFile("./shaders/convolutional_train_forward.glsl");
  const std::string backpropSrc = loadFile("./shaders/convolutional_backprop.glsl");
  const std::string updateParamsSrc = loadFile("./shaders/convolutional_update_params.glsl");

  m_evalForwardShader = m_gpu.compileShader(evalForwardSrc, evalForwardBuffers,
    evalForwardConstants, workgroupSize, numWorkgroups, includesDir);
  m_trainForwardShader = m_gpu.compileShader(trainForwardSrc, trainForwardBuffers,
    trainForwardConstants, workgroupSize, numWorkgroups, includesDir);
  m_backpropShader = m_gpu.compileShader(backpropSrc, backpropBuffers, backpropConstants,
    workgroupSize, numWorkgroups, includesDir);
  m_updateParamsShader = m_gpu.compileShader(updateParamsSrc, updateParamsBuffers,
    updateParamsConstants, workgroupSize, numWorkgroups, includesDir);
}

size_t ConvolutionalLayer::size() const {
  return calcProduct(outputSize());
}

Size3 ConvolutionalLayer::outputSize() const {
  return {
    m_inputW - m_kernelSize[0] + 1,
    m_inputH - m_kernelSize[1] + 1,
    m_depth
  };
}

void ConvolutionalLayer::evalForward() {
  m_gpu.queueShader(m_evalForwardShader);
}

void ConvolutionalLayer::trainForward() {
  m_gpu.queueShader(m_trainForwardShader);
}

void ConvolutionalLayer::backprop() {
  m_gpu.queueShader(m_backpropShader);
}

void ConvolutionalLayer::updateParams() {
  m_gpu.queueShader(m_updateParamsShader);
}

GpuBufferHandle ConvolutionalLayer::outputBuffer() const {
  return m_bufferA.handle;
}

GpuBufferHandle ConvolutionalLayer::weightsBuffer() const {
  return m_bufferK.handle;
}

GpuBufferHandle ConvolutionalLayer::deltaBuffer() const {
  return m_bufferD.handle;
}

void ConvolutionalLayer::retrieveBuffers() {
  m_gpu.retrieveBuffer(m_bufferK.handle, m_kernelData.data());
  m_gpu.retrieveBuffer(m_bufferB.handle, m_biasData.data());
}

void ConvolutionalLayer::writeToStream(std::ostream& stream) const {
  size_t kernelSize = m_kernelSize[0] * m_kernelSize[1] * m_inputDepth;

  for (size_t i = 0; i < m_depth; ++i) {
    stream.write(reinterpret_cast<const char*>(m_biasData.data() + i), sizeof(netfloat_t));
    stream.write(reinterpret_cast<const char*>(m_kernelData.data() + i * kernelSize),
      kernelSize * sizeof(netfloat_t));
  }
}

}
}
