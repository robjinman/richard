#include "gpu/output_layer.hpp"
#include "utils.hpp"
#include "file_system.hpp"
#include "platform_paths.hpp"
#include "config.hpp"
#include <cstring>

namespace richard {
namespace gpu {

OutputLayer::OutputLayer(Gpu& gpu, FileSystem& fileSystem, const PlatformPaths& platformPaths,
  const Config& config, std::istream& stream, size_t inputSize)
  : m_gpu(gpu)
  , m_fileSystem(fileSystem)
  , m_platformPaths(platformPaths) {

  initialize(config, inputSize);

  stream.read(reinterpret_cast<char*>(m_B.data()), m_size * sizeof(netfloat_t));
  stream.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

OutputLayer::OutputLayer(Gpu& gpu, FileSystem& fileSystem, const PlatformPaths& platformPaths,
  const Config& config, size_t inputSize)
  : m_gpu(gpu)
  , m_fileSystem(fileSystem)
  , m_platformPaths(platformPaths) {

  initialize(config, inputSize);

  m_W.randomize(0.1);
}

void OutputLayer::initialize(const Config& config, size_t inputSize) {
  m_inputSize = inputSize;

  m_size = config.getInteger("size");
  m_learnRate = config.getFloat("learnRate");
  m_learnRateDecay = config.getFloat("learnRateDecay");

  m_B = Vector(m_size);
  m_W = Matrix(m_inputSize, m_size);
  m_A = Vector(m_size);
}

void OutputLayer::allocateGpuBuffers() {
  GpuBufferFlags paramBuffersFlags = GpuBufferFlags::large
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;

  GpuBufferFlags activationsBufferFlags = GpuBufferFlags::large
                                        | GpuBufferFlags::hostReadAccess
                                        | GpuBufferFlags::frequentHostAccess;

  m_bufferB = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), paramBuffersFlags);
  m_bufferW = m_gpu.allocateBuffer(m_inputSize * m_size * sizeof(netfloat_t), paramBuffersFlags);
  m_bufferZ = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), GpuBufferFlags::large);
  m_bufferA = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), activationsBufferFlags);
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

void OutputLayer::createGpuShaders(GpuBufferHandle inputBuffer, GpuBufferHandle statusBuffer,
  const Layer*, GpuBufferHandle sampleYBuffer) {

  createEvalForwardShader(inputBuffer);
  createTrainForwardShader(inputBuffer);
  createBackpropDeltaShader(statusBuffer, inputBuffer, sampleYBuffer);
  createBackpropInputDeltaShader();
  createUpdateParamsShader(statusBuffer);
}

void OutputLayer::createEvalForwardShader(GpuBufferHandle inputBuffer) {
  GpuBufferBindings buffers{
    { inputBuffer, BufferAccessMode::read },
    { m_bufferB.handle, BufferAccessMode::read },
    { m_bufferW.handle, BufferAccessMode::read },
    { m_bufferA.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) }
  };

  const std::string sourceName = "output_eval_forward.glsl";
  const std::string source = m_fileSystem.loadTextFile(m_platformPaths.get("shaders", sourceName));

  Size3 workSize{ m_size, 1, 1 };

  m_evalForwardShader = m_gpu.compileShader(sourceName, source, buffers, constants, workSize,
    m_platformPaths.get("shaders"));
}

void OutputLayer::createTrainForwardShader(GpuBufferHandle inputBuffer) {
  GpuBufferBindings buffers{
    { inputBuffer, BufferAccessMode::read },
    { m_bufferB.handle, BufferAccessMode::read },
    { m_bufferW.handle, BufferAccessMode::read },
    { m_bufferZ.handle, BufferAccessMode::write },
    { m_bufferA.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) }
  };

  const std::string sourceName = "output_train_forward.glsl";
  const std::string source = m_fileSystem.loadTextFile(m_platformPaths.get("shaders", sourceName));

  Size3 workSize{ m_size, 1, 1 };

  m_trainForwardShader = m_gpu.compileShader(sourceName, source, buffers, constants, workSize,
    m_platformPaths.get("shaders"));
}

void OutputLayer::createBackpropDeltaShader(GpuBufferHandle statusBuffer,
  GpuBufferHandle inputBuffer, GpuBufferHandle sampleYBuffer) {

  GpuBufferBindings buffers{
    { statusBuffer, BufferAccessMode::read },
    { inputBuffer, BufferAccessMode::read },
    { sampleYBuffer, BufferAccessMode::read },
    { m_bufferB.handle, BufferAccessMode::read },
    { m_bufferW.handle, BufferAccessMode::read },
    { m_bufferZ.handle, BufferAccessMode::read },
    { m_bufferA.handle, BufferAccessMode::read },
    { m_bufferD.handle, BufferAccessMode::write },
    { m_bufferDeltaB.handle, BufferAccessMode::write },
    { m_bufferDeltaW.handle, BufferAccessMode::write }
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) }
  };

  const std::string sourceName = "output_backprop_delta.glsl";
  const std::string source = m_fileSystem.loadTextFile(m_platformPaths.get("shaders", sourceName));

  Size3 workSize{ m_size, 1, 1 };

  m_backpropDeltaShader = m_gpu.compileShader(sourceName, source, buffers, constants, workSize,
    m_platformPaths.get("shaders"));
}

void OutputLayer::createBackpropInputDeltaShader() {
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

void OutputLayer::createUpdateParamsShader(GpuBufferHandle statusBuffer) {
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

size_t OutputLayer::size() const {
  return m_size;
}

Size3 OutputLayer::outputSize() const {
  return { m_size, 1, 1 };
}

const Vector& OutputLayer::activations() const {
  memcpy(m_A.data(), m_bufferA.data, m_bufferA.size);
  return m_A;
}

void OutputLayer::evalForward() {
  m_gpu.queueShader(m_evalForwardShader);
}

void OutputLayer::trainForward() {
  m_gpu.queueShader(m_trainForwardShader);
}

void OutputLayer::backprop() {
  m_gpu.queueShader(m_backpropDeltaShader);
  m_gpu.queueShader(m_backpropInputDeltaShader);
}

void OutputLayer::updateParams() {
  m_gpu.queueShader(m_updateParamsShader);
}

GpuBufferHandle OutputLayer::outputBuffer() const {
  return m_bufferA.handle;
}

GpuBufferHandle OutputLayer::weightsBuffer() const {
  return m_bufferW.handle;
}

GpuBufferHandle OutputLayer::deltaBuffer() const {
  return m_bufferD.handle;
}

GpuBufferHandle OutputLayer::inputDeltaBuffer() const {
  return m_bufferInputDelta.handle;
}

void OutputLayer::retrieveBuffers() {
  m_gpu.retrieveBuffer(m_bufferB.handle, m_B.data());
  m_gpu.retrieveBuffer(m_bufferW.handle, m_W.data());
}

void OutputLayer::writeToStream(std::ostream& stream) const {
  stream.write(reinterpret_cast<const char*>(m_B.data()), m_B.size() * sizeof(netfloat_t));
  stream.write(reinterpret_cast<const char*>(m_W.data()),
    m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

void OutputLayer::test_setWeights(const DataArray& W) {
  m_W = Matrix(W, m_W.cols(), m_W.rows());
}

void OutputLayer::test_setBiases(const DataArray& B) {
  m_B = B;
}

GpuBufferHandle OutputLayer::test_deltaWBuffer() const {
  return m_bufferDeltaW.handle;
}

GpuBufferHandle OutputLayer::test_deltaBBuffer() const {
  return m_bufferDeltaB.handle;
}

const Matrix& OutputLayer::test_W() const {
  return m_W;
}

const Vector& OutputLayer::test_B() const {
  return m_B;
}

}
}
