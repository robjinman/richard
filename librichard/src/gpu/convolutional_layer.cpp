#include "richard/gpu/convolutional_layer.hpp"
#include "richard/utils.hpp"
#include "richard/math.hpp"
#include "richard/file_system.hpp"
#include "richard/platform_paths.hpp"
#include "richard/config.hpp"

namespace richard {
namespace gpu {

ConvolutionalLayer::ConvolutionalLayer(Gpu& gpu, FileSystem& fileSystem,
  const PlatformPaths& platformPaths, const Config& config, const Size3& inputShape,
  bool isFirstLayer)
  : m_gpu(gpu)
  , m_fileSystem(fileSystem)
  , m_platformPaths(platformPaths) {

  initialize(config, inputShape, isFirstLayer);
}

ConvolutionalLayer::ConvolutionalLayer(Gpu& gpu, FileSystem& fileSystem,
  const PlatformPaths& platformPaths, const Config& config, std::istream& stream,
  const Size3& inputShape, bool isFirstLayer)
  : m_gpu(gpu)
  , m_fileSystem(fileSystem)
  , m_platformPaths(platformPaths) {

  initialize(config, inputShape, isFirstLayer);

  size_t kernelSize = m_kernelSize[0] * m_kernelSize[1] * m_inputDepth;

  for (size_t i = 0; i < m_depth; ++i) {
    stream.read(reinterpret_cast<char*>(m_biasData.data() + i), sizeof(netfloat_t));
    stream.read(reinterpret_cast<char*>(m_kernelData.data() + i * kernelSize),
      kernelSize * sizeof(netfloat_t));
  }
}

void ConvolutionalLayer::initialize(const Config& config, const Size3& inputShape,
  bool isFirstLayer) {

  m_inputW = inputShape[0];
  m_inputH = inputShape[1];
  m_inputDepth = inputShape[2];
  m_kernelSize = config.getNumberArray<size_t, 2>("kernelSize");
  m_depth = config.getNumber<size_t>("depth");
  m_learnRate = config.getNumber<netfloat_t>("learnRate");
  m_learnRateDecay = config.getNumber<netfloat_t>("learnRateDecay");
  m_dropoutRate = config.getNumber<netfloat_t>("dropoutRate");
  m_isFirstLayer = isFirstLayer;
  m_kernelData = Vector(m_kernelSize[0] * m_kernelSize[1] * m_inputDepth * m_depth);
  m_biasData = Vector(m_depth);

  Size3 kernelShape{ m_kernelSize[0], m_kernelSize[1], m_inputDepth };
  size_t kernelSize = calcProduct(kernelShape);
  for (size_t i = 0; i < m_depth; ++i) {
    KernelPtr kernel = Kernel::createShallow(m_kernelData.data() + i * kernelSize, kernelShape);
    kernel->randomize(0.1f);
  }

  ASSERT_MSG(m_kernelSize[0] <= m_inputW,
    "Kernel width " << m_kernelSize[0] << " is larger than input width " << m_inputW);

  ASSERT_MSG(m_kernelSize[1] <= m_inputH,
    "Kernel height " << m_kernelSize[1] << " is larger than input height " << m_inputH);
}

void ConvolutionalLayer::allocateGpuBuffers() {
  size_t kernelSize = m_kernelSize[0] * m_kernelSize[1] * m_inputDepth;
  size_t featureMapSizeBytes = calcProduct(outputSize()) * sizeof(netfloat_t);
  size_t inputSizeBytes = m_inputW * m_inputH * m_inputDepth * sizeof(netfloat_t);

  GpuBufferFlags paramBuffersFlags = GpuBufferFlags::large
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;

  m_bufferK = m_gpu.allocateBuffer(m_depth * kernelSize * sizeof(netfloat_t), paramBuffersFlags);
  m_bufferB = m_gpu.allocateBuffer(m_depth * sizeof(netfloat_t), paramBuffersFlags);
  m_bufferZ = m_gpu.allocateBuffer(featureMapSizeBytes, GpuBufferFlags::large);
  m_bufferA = m_gpu.allocateBuffer(featureMapSizeBytes, GpuBufferFlags::large);
  m_bufferD = m_gpu.allocateBuffer(featureMapSizeBytes, GpuBufferFlags::large);
  m_bufferInputDelta = m_gpu.allocateBuffer(inputSizeBytes, GpuBufferFlags::large);
  m_bufferDeltaK = m_gpu.allocateBuffer(m_depth * kernelSize * sizeof(netfloat_t),
    GpuBufferFlags::large | GpuBufferFlags::hostWriteAccess);
  m_bufferDeltaB = m_gpu.allocateBuffer(m_depth * sizeof(netfloat_t),
    GpuBufferFlags::large | GpuBufferFlags::hostWriteAccess);

  m_gpu.submitBufferData(m_bufferK.handle, m_kernelData.data());

  Vector deltaKData(m_kernelData.size());
  m_gpu.submitBufferData(m_bufferDeltaK.handle, deltaKData.data());

  Vector deltaBData(m_biasData.size());
  m_gpu.submitBufferData(m_bufferDeltaB.handle, deltaBData.data());

  m_gpu.submitBufferData(m_bufferB.handle, m_biasData.data());
}

void ConvolutionalLayer::createGpuShaders(GpuBufferHandle inputBuffer, GpuBufferHandle statusBuffer,
  const Layer* nextLayer, GpuBufferHandle) {

  DBG_ASSERT(nextLayer != nullptr);

  createEvalForwardShader(inputBuffer);
  createTrainForwardShader(statusBuffer, inputBuffer);
  createBackpropDeltaShader(nextLayer);
  createBackpropInputDeltaShader();
  createBackpropParamDeltasShader(statusBuffer, inputBuffer);
  createUpdateParamsShader(statusBuffer);
}

void ConvolutionalLayer::createEvalForwardShader(GpuBufferHandle inputBuffer) {
  GpuBufferBindings buffers{
    { inputBuffer, BufferAccessMode::read },
    { m_bufferK.handle, BufferAccessMode::read },
    { m_bufferB.handle, BufferAccessMode::read },
    { m_bufferA.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputDepth) }
  };

  std::string shaderName = "convolutional_eval_forward.spv";
  auto shaderCode = m_fileSystem.loadBinaryFile(m_platformPaths.get("shaders", shaderName));

  Size3 workSize{ outputSize()[0], outputSize()[1], m_depth };

  m_evalForwardShader = m_gpu.addShader(shaderName, shaderCode, buffers, constants, 0, workSize);
}

void ConvolutionalLayer::createTrainForwardShader(GpuBufferHandle statusBuffer,
  GpuBufferHandle inputBuffer) {

  GpuBufferBindings buffers{
    { statusBuffer, BufferAccessMode::read },
    { inputBuffer, BufferAccessMode::read },
    { m_bufferK.handle, BufferAccessMode::read },
    { m_bufferB.handle, BufferAccessMode::read },
    { m_bufferZ.handle, BufferAccessMode::write },
    { m_bufferA.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputDepth) },
    { SpecializationConstant::Type::bool_type, m_isFirstLayer },
    { SpecializationConstant::Type::float_type, m_dropoutRate }
  };

  std::string shaderName = "convolutional_train_forward.spv";
  auto shaderCode = m_fileSystem.loadBinaryFile(m_platformPaths.get("shaders", shaderName));

  Size3 workSize{ outputSize()[0], outputSize()[1], m_depth };

  m_trainForwardShader = m_gpu.addShader(shaderName, shaderCode, buffers, constants,
    sizeof(uint32_t), workSize);
}

void ConvolutionalLayer::createBackpropDeltaShader(const Layer* nextLayer) {
  GpuBufferBindings buffers{
    { m_bufferZ.handle, BufferAccessMode::read },
    { m_bufferD.handle, BufferAccessMode::write },
    { nextLayer->inputDeltaBuffer(), BufferAccessMode::read }
  };

  std::string shaderName = "convolutional_backprop_delta.spv";
  auto shaderCode = m_fileSystem.loadBinaryFile(m_platformPaths.get("shaders", shaderName));

  Size3 workSize{ outputSize()[0], outputSize()[1], m_depth };

  m_backpropDeltaShader = m_gpu.addShader(shaderName, shaderCode, buffers, {}, 0, workSize);
}

void ConvolutionalLayer::createBackpropInputDeltaShader() {
  GpuBufferBindings buffers{
    { m_bufferK.handle, BufferAccessMode::read },
    { m_bufferD.handle, BufferAccessMode::read },
    { m_bufferInputDelta.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputDepth) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_depth) }
  };

  std::string shaderName = "convolutional_backprop_input_delta.spv";
  auto shaderCode = m_fileSystem.loadBinaryFile(m_platformPaths.get("shaders", shaderName));

  Size3 workSize{ m_inputW, m_inputH, m_inputDepth };

  m_backpropInputDeltaShader = m_gpu.addShader(shaderName, shaderCode, buffers, constants, 0,
    workSize);
}

void ConvolutionalLayer::createBackpropParamDeltasShader(GpuBufferHandle statusBuffer,
  GpuBufferHandle inputBuffer) {

  GpuBufferBindings buffers{
    { statusBuffer, BufferAccessMode::read },
    { inputBuffer, BufferAccessMode::read },
    { m_bufferD.handle, BufferAccessMode::read },
    { m_bufferDeltaK.handle, BufferAccessMode::write },
    { m_bufferDeltaB.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(outputSize()[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(outputSize()[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputW) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputH) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputDepth) },
    { SpecializationConstant::Type::bool_type, m_isFirstLayer }
  };

  std::string shaderName = "convolutional_backprop_param_deltas.spv";
  auto shaderCode = m_fileSystem.loadBinaryFile(m_platformPaths.get("shaders", shaderName));

  Size3 workSize{ m_kernelSize[0] * m_kernelSize[1], m_inputDepth, m_depth };

  m_backpropParamDeltasShader = m_gpu.addShader(shaderName, shaderCode, buffers, constants, 0,
    workSize);
}

void ConvolutionalLayer::createUpdateParamsShader(GpuBufferHandle statusBuffer) {
  GpuBufferBindings buffers{
    { statusBuffer, BufferAccessMode::read },
    { m_bufferK.handle, BufferAccessMode::write },
    { m_bufferB.handle, BufferAccessMode::write },
    { m_bufferDeltaK.handle, BufferAccessMode::write },
    { m_bufferDeltaB.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[0]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_kernelSize[1]) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputDepth) },
    { SpecializationConstant::Type::float_type, m_learnRate },
    { SpecializationConstant::Type::float_type, m_learnRateDecay }
  };

  std::string shaderName = "convolutional_update_params.spv";
  auto shaderCode = m_fileSystem.loadBinaryFile(m_platformPaths.get("shaders", shaderName));

  Size3 workSize{ m_kernelSize[0] * m_kernelSize[1], m_inputDepth, m_depth };

  m_updateParamsShader = m_gpu.addShader(shaderName, shaderCode, buffers, constants, 0, workSize);
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
  uint32_t seed = static_cast<uint32_t>(rand());
  m_gpu.queueShader(m_trainForwardShader, &seed);
}

void ConvolutionalLayer::backprop() {
  m_gpu.queueShader(m_backpropDeltaShader);
  m_gpu.queueShader(m_backpropInputDeltaShader);
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
