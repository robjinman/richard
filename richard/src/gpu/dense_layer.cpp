#include "gpu/dense_layer.hpp"
#include "utils.hpp"
#include "file_system.hpp"
#include "platform_paths.hpp"
#include "config.hpp"

namespace richard {
namespace gpu {

DenseLayer::DenseLayer(Gpu& gpu, FileSystem& fileSystem, const PlatformPaths& platformPaths,
  const Config& config, size_t inputSize, bool isFirstLayer)
  : m_gpu(gpu)
  , m_fileSystem(fileSystem)
  , m_platformPaths(platformPaths) {

  initialize(config, inputSize, isFirstLayer);
}

DenseLayer::DenseLayer(Gpu& gpu, FileSystem& fileSystem, const PlatformPaths& platformPaths,
  const Config& config, std::istream& stream, size_t inputSize, bool isFirstLayer)
  : m_gpu(gpu)
  , m_fileSystem(fileSystem)
  , m_platformPaths(platformPaths) {

  initialize(config, inputSize, isFirstLayer);

  stream.read(reinterpret_cast<char*>(m_B.data()), m_size * sizeof(netfloat_t));
  stream.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

void DenseLayer::initialize(const Config& config, size_t inputSize, bool isFirstLayer) {
  m_inputSize = inputSize;
  m_isFirstLayer = isFirstLayer;

  m_size = config.getValue<size_t>("size");
  m_learnRate = config.getValue<netfloat_t>("learnRate");
  m_learnRateDecay = config.getValue<netfloat_t>("learnRateDecay");
  m_dropoutRate = config.getValue<netfloat_t>("dropoutRate");

  m_B = Vector(m_size);
  m_W = Matrix(m_inputSize, m_size);

  m_W.randomize(0.1);
}

void DenseLayer::allocateGpuBuffers() {
  GpuBufferFlags paramBuffersFlags = GpuBufferFlags::large
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;

  m_bufferB = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), paramBuffersFlags);
  m_bufferW = m_gpu.allocateBuffer(m_inputSize * m_size * sizeof(netfloat_t),
    paramBuffersFlags);
  m_bufferZ = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), GpuBufferFlags::large);
  m_bufferA = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), GpuBufferFlags::large);
  m_bufferD = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), GpuBufferFlags::large);
  m_bufferInputDelta = m_gpu.allocateBuffer(m_inputSize * sizeof(netfloat_t),
    GpuBufferFlags::large);
  m_bufferDeltaB = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t),
    GpuBufferFlags::large | GpuBufferFlags::hostWriteAccess);
  m_bufferDeltaW = m_gpu.allocateBuffer(m_inputSize * m_size * sizeof(netfloat_t),
    GpuBufferFlags::large | GpuBufferFlags::hostWriteAccess);

  m_gpu.submitBufferData(m_bufferB.handle, m_B.data());
  m_gpu.submitBufferData(m_bufferW.handle, m_W.data());

  Matrix deltaW(m_W.cols(), m_W.rows());
  m_gpu.submitBufferData(m_bufferDeltaW.handle, deltaW.data());

  Vector deltaB(m_B.size());
  m_gpu.submitBufferData(m_bufferDeltaB.handle, deltaB.data());
}

void DenseLayer::createGpuShaders(GpuBufferHandle inputBuffer, GpuBufferHandle statusBuffer,
  const Layer* nextLayer, GpuBufferHandle) {

  DBG_ASSERT(nextLayer != nullptr);

  createEvalForwardShader(inputBuffer);
  createTrainForwardShader(statusBuffer, inputBuffer);
  createBackpropDeltaShader(statusBuffer, inputBuffer, nextLayer);
  createBackpropInputDeltaShader();
  createUpdateParamsShader(statusBuffer);
}

void DenseLayer::createEvalForwardShader(GpuBufferHandle inputBuffer) {
  GpuBufferBindings buffers{
    { inputBuffer, BufferAccessMode::read },
    { m_bufferB.handle, BufferAccessMode::read },
    { m_bufferW.handle, BufferAccessMode::read },
    { m_bufferA.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) }
  };

  const std::string sourceName = "dense_eval_forward.glsl";
  const std::string source = m_fileSystem.loadTextFile(m_platformPaths.get("shaders", sourceName));

  Size3 workSize{ m_size, 1, 1 };

  m_evalForwardShader = m_gpu.compileShader(sourceName, source, buffers, constants, workSize,
    m_platformPaths.get("shaders"));
}

void DenseLayer::createTrainForwardShader(GpuBufferHandle statusBuffer,
  GpuBufferHandle inputBuffer) {

  GpuBufferBindings buffers{
    { statusBuffer, BufferAccessMode::read },
    { inputBuffer, BufferAccessMode::read },
    { m_bufferB.handle, BufferAccessMode::read },
    { m_bufferW.handle, BufferAccessMode::read },
    { m_bufferZ.handle, BufferAccessMode::write },
    { m_bufferA.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) },
    { SpecializationConstant::Type::bool_type, m_isFirstLayer },
  //  { SpecializationConstant::Type::float_type, m_dropoutRate }
  };

  const std::string sourceName = "dense_train_forward.glsl";
  const std::string source = m_fileSystem.loadTextFile(m_platformPaths.get("shaders", sourceName));

  Size3 workSize{ m_size, 1, 1 };

  m_trainForwardShader = m_gpu.compileShader(sourceName, source, buffers, constants, workSize,
    m_platformPaths.get("shaders"));
}

void DenseLayer::createBackpropDeltaShader(GpuBufferHandle statusBuffer,
  GpuBufferHandle inputBuffer, const Layer* nextLayer) {

  GpuBufferBindings buffers{
    { statusBuffer, BufferAccessMode::read },
    { inputBuffer, BufferAccessMode::read },
    { m_bufferB.handle, BufferAccessMode::read },
    { m_bufferW.handle, BufferAccessMode::read },
    { m_bufferZ.handle, BufferAccessMode::read },
    { m_bufferA.handle, BufferAccessMode::read },
    { m_bufferD.handle, BufferAccessMode::write },
    { nextLayer->weightsBuffer(), BufferAccessMode::read },
    { nextLayer->deltaBuffer(), BufferAccessMode::read },
    { m_bufferDeltaB.handle, BufferAccessMode::write },
    { m_bufferDeltaW.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(nextLayer->size()) },
    { SpecializationConstant::Type::bool_type, m_isFirstLayer },
  };

  const std::string sourceName = "dense_backprop_delta.glsl";
  const std::string source = m_fileSystem.loadTextFile(m_platformPaths.get("shaders", sourceName));

  Size3 workSize{ m_size, 1, 1 };

  m_backpropDeltaShader = m_gpu.compileShader(sourceName, source, buffers, constants, workSize,
    m_platformPaths.get("shaders"));
}

void DenseLayer::createBackpropInputDeltaShader() {
  GpuBufferBindings buffers{
    { m_bufferW.handle, BufferAccessMode::read },
    { m_bufferD.handle, BufferAccessMode::read },
    { m_bufferInputDelta.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_size) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) }
  };

  const std::string sourceName = "dense_backprop_input_delta.glsl";
  const std::string source = m_fileSystem.loadTextFile(m_platformPaths.get("shaders", sourceName));

  Size3 workSize{ m_inputSize, 1, 1 };

  m_backpropInputDeltaShader = m_gpu.compileShader(sourceName, source, buffers, constants, workSize,
    m_platformPaths.get("shaders"));
}

void DenseLayer::createUpdateParamsShader(GpuBufferHandle statusBuffer) {
  GpuBufferBindings buffers{
    { statusBuffer, BufferAccessMode::read },
    { m_bufferB.handle, BufferAccessMode::write },
    { m_bufferW.handle, BufferAccessMode::write },
    { m_bufferDeltaB.handle, BufferAccessMode::write },
    { m_bufferDeltaW.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) },
    { SpecializationConstant::Type::float_type, m_learnRate },
    { SpecializationConstant::Type::float_type, m_learnRateDecay },
  };

  const std::string sourceName = "dense_update_params.glsl";
  const std::string source = m_fileSystem.loadTextFile(m_platformPaths.get("shaders", sourceName));

  Size3 workSize{ m_inputSize, m_size, 1 };

  m_updateParamsShader = m_gpu.compileShader(sourceName, source, buffers, constants, workSize,
    m_platformPaths.get("shaders"));
}

size_t DenseLayer::size() const {
  return m_size;
}

Size3 DenseLayer::outputSize() const {
  return { m_size, 1, 1 };
}

void DenseLayer::evalForward() {
  m_gpu.queueShader(m_evalForwardShader);
}

void DenseLayer::trainForward() {
  m_gpu.queueShader(m_trainForwardShader);
}

void DenseLayer::backprop() {
  m_gpu.queueShader(m_backpropDeltaShader);
  m_gpu.queueShader(m_backpropInputDeltaShader);
}

void DenseLayer::updateParams() {
  m_gpu.queueShader(m_updateParamsShader);
}

GpuBufferHandle DenseLayer::outputBuffer() const {
  return m_bufferA.handle;
}

GpuBufferHandle DenseLayer::weightsBuffer() const {
  return m_bufferW.handle;
}

GpuBufferHandle DenseLayer::deltaBuffer() const {
  return m_bufferD.handle;
}

GpuBufferHandle DenseLayer::inputDeltaBuffer() const {
  return m_bufferInputDelta.handle;
}

void DenseLayer::retrieveBuffers() {
  m_gpu.retrieveBuffer(m_bufferB.handle, m_B.data());
  m_gpu.retrieveBuffer(m_bufferW.handle, m_W.data());
}

void DenseLayer::writeToStream(std::ostream& stream) const {
  stream.write(reinterpret_cast<const char*>(m_B.data()), m_B.size() * sizeof(netfloat_t));
  stream.write(reinterpret_cast<const char*>(m_W.data()),
    m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

void DenseLayer::test_setWeights(const DataArray& W) {
  m_W = Matrix(W, m_W.cols(), m_W.rows());
}

void DenseLayer::test_setBiases(const DataArray& B) {
  m_B = B;
}

GpuBufferHandle DenseLayer::test_deltaWBuffer() const {
  return m_bufferDeltaW.handle;
}

GpuBufferHandle DenseLayer::test_deltaBBuffer() const {
  return m_bufferDeltaB.handle;
}

const Matrix& DenseLayer::test_W() const {
  return m_W;
}

const Vector& DenseLayer::test_B() const {
  return m_B;
}

}
}
