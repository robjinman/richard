#include "gpu/convolutional_layer.hpp"
#include "gpu/gpu_utils.hpp"
#include "utils.hpp"
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

  ASSERT_MSG(m_kernelSize[0] <= m_inputW,
    "Kernel width " << m_kernelSize[0] << " is larger than input width " << m_inputW);

  ASSERT_MSG(m_kernelSize[1] <= m_inputH,
    "Kernel height " << m_kernelSize[1] << " is larger than input height " << m_inputH);
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
  m_bufferInputDelta = m_gpu.allocateBuffer(featureMapSizeBytes, GpuBufferFlags::large);
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

  createEvalForwardShader(inputBuffer);
  createTrainForwardShader(statusBuffer, inputBuffer);
  createBackpropDeltaShader(nextLayer);
  createBackpropParamDeltasShader(statusBuffer, inputBuffer);
  createUpdateParamsShader(statusBuffer);
}

void ConvolutionalLayer::createEvalForwardShader(GpuBufferHandle inputBuffer) {
  GpuBufferBindings buffers{
    inputBuffer,
    m_bufferK.handle,
    m_bufferB.handle,
    m_bufferA.handle
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_depth) }
  };

  Size3 workgroupSize;
  Size3 numWorkgroups;
  optimumWorkgroups({ outputSize()[0], outputSize()[1], m_depth }, workgroupSize, numWorkgroups);

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string source = loadFile("./shaders/convolutional_eval_forward.glsl");

  m_evalForwardShader = m_gpu.compileShader(source, buffers, constants, workgroupSize,
    numWorkgroups, includesDir);
}

void ConvolutionalLayer::createTrainForwardShader(GpuBufferHandle statusBuffer,
  GpuBufferHandle inputBuffer) {

  GpuBufferBindings buffers{
    statusBuffer,
    inputBuffer,
    m_bufferK.handle,
    m_bufferB.handle,
    m_bufferZ.handle,
    m_bufferA.handle
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_depth) },
    { SpecializationConstant::Type::bool_type, m_isFirstLayer },
  //  { SpecializationConstant::Type::float_type, m_dropoutRate }
  };

  Size3 workgroupSize;
  Size3 numWorkgroups;
  optimumWorkgroups({ outputSize()[0], outputSize()[1], m_depth }, workgroupSize, numWorkgroups);

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string source = loadFile("./shaders/convolutional_train_forward.glsl");

  m_trainForwardShader = m_gpu.compileShader(source, buffers, constants, workgroupSize,
    numWorkgroups, includesDir);
}

void ConvolutionalLayer::createBackpropDeltaShader(const Layer* nextLayer) {
  GpuBufferBindings buffers{
    m_bufferZ.handle,
    m_bufferD.handle,
    nextLayer->inputDeltaBuffer()
  };

  Size3 workgroupSize;
  Size3 numWorkgroups;
  optimumWorkgroups({ outputSize()[0], outputSize()[1], m_depth }, workgroupSize, numWorkgroups);

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string source = loadFile("./shaders/convolutional_backprop_delta.glsl");

  m_backpropDeltaShader = m_gpu.compileShader(source, buffers, {}, workgroupSize, numWorkgroups,
    includesDir);
}

void ConvolutionalLayer::createBackpropInputDeltaShader() {
  GpuBufferBindings buffers{
    m_bufferK.handle,
    m_bufferD.handle,
    m_bufferInputDelta.handle
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[2]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_depth) }
  };

  Size3 workgroupSize;
  Size3 numWorkgroups;
  optimumWorkgroups({ m_inputW, m_inputH, m_inputDepth }, workgroupSize, numWorkgroups);

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string source = loadFile("./shaders/convolutional_backprop_input_delta.glsl");

  m_backpropInputDeltaShader = m_gpu.compileShader(source, buffers, constants, workgroupSize,
    numWorkgroups, includesDir);
}

void ConvolutionalLayer::createBackpropParamDeltasShader(GpuBufferHandle statusBuffer,
  GpuBufferHandle inputBuffer) {

  GpuBufferBindings buffers{
    statusBuffer,
    inputBuffer,
    m_bufferD.handle,
    m_bufferDeltaK.handle,
    m_bufferDeltaB.handle
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(outputSize()[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(outputSize()[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputW) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputH) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputDepth) },
    { SpecializationConstant::Type::bool_type, m_isFirstLayer }
  };

  Size3 workgroupSize;
  Size3 numWorkgroups;
  optimumWorkgroups({ m_kernelSize[0] * m_kernelSize[1], m_inputDepth, m_depth },
    workgroupSize, numWorkgroups);

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string source = loadFile("./shaders/convolutional_backprop_param_deltas.glsl");

  m_backpropParamDeltasShader = m_gpu.compileShader(source, buffers, constants, workgroupSize,
    numWorkgroups, includesDir);
}

void ConvolutionalLayer::createUpdateParamsShader(GpuBufferHandle statusBuffer) {
  GpuBufferBindings buffers{
    statusBuffer,
    m_bufferK.handle,
    m_bufferB.handle,
    m_bufferDeltaK.handle,
    m_bufferDeltaB.handle
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputDepth) },
    { SpecializationConstant::Type::float_type, m_learnRate },
    { SpecializationConstant::Type::float_type, m_learnRateDecay }
  };

  Size3 workgroupSize;
  Size3 numWorkgroups;
  optimumWorkgroups({ m_kernelSize[0] * m_kernelSize[1], m_inputDepth, m_depth }, workgroupSize,
    numWorkgroups);

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string source = loadFile("./shaders/convolutional_update_params.glsl");

  m_updateParamsShader = m_gpu.compileShader(source, buffers, constants, workgroupSize,
    numWorkgroups, includesDir);
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
  m_gpu.queueShader(m_backpropDeltaShader);
  m_gpu.queueShader(m_backpropParamDeltasShader);
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

GpuBufferHandle ConvolutionalLayer::inputDeltaBuffer() const {
  return m_bufferInputDelta.handle;
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

void ConvolutionalLayer::test_setKernels(const DataArray& kernelData) {
  m_kernelData = kernelData;
}

void ConvolutionalLayer::test_setBiases(const DataArray& biasData) {
  m_biasData = biasData;
}

GpuBufferHandle ConvolutionalLayer::test_activationsBuffer() const {
  return m_bufferA.handle;
}

GpuBufferHandle ConvolutionalLayer::test_deltaKBuffer() const {
  return m_bufferDeltaK.handle;
}

GpuBufferHandle ConvolutionalLayer::test_deltaBBuffer() const {
  return m_bufferDeltaB.handle;
}

const DataArray& ConvolutionalLayer::test_kernels() const {
  return m_kernelData.storage();
}

const Vector& ConvolutionalLayer::test_biases() const {
  return m_biasData;
}

}
}
