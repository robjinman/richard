#include "gpu/dense_layer.hpp"
#include "utils.hpp"

namespace richard {
namespace gpu {

DenseLayer::DenseLayer(Gpu& gpu, const nlohmann::json& obj, size_t inputSize, bool isFirstLayer)
  : m_gpu(gpu) {

  initialize(obj, inputSize, isFirstLayer);
}

DenseLayer::DenseLayer(Gpu& gpu, const nlohmann::json& obj, std::istream& stream, size_t inputSize,
  bool isFirstLayer)
  : m_gpu(gpu) {

  initialize(obj, inputSize, isFirstLayer);

  stream.read(reinterpret_cast<char*>(m_B.data()), m_size * sizeof(netfloat_t));
  stream.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

void DenseLayer::initialize(const nlohmann::json& obj, size_t inputSize, bool isFirstLayer) {
  m_inputSize = inputSize;
  m_isFirstLayer = isFirstLayer;

  m_size = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<netfloat_t>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<netfloat_t>();
  m_dropoutRate = getOrThrow(obj, "dropoutRate").get<netfloat_t>();

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
    inputBuffer,
    m_bufferB.handle,
    m_bufferW.handle,
    m_bufferA.handle
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) }
  };

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string source = loadFile("./shaders/dense_eval_forward.glsl");

  Size3 workSize{ static_cast<uint32_t>(m_size), 1, 1 };

  m_evalForwardShader = m_gpu.compileShader(source, buffers, constants, workSize, includesDir);
}

void DenseLayer::createTrainForwardShader(GpuBufferHandle statusBuffer,
  GpuBufferHandle inputBuffer) {

  GpuBufferBindings buffers{
    statusBuffer,
    inputBuffer,
    m_bufferB.handle,
    m_bufferW.handle,
    m_bufferZ.handle,
    m_bufferA.handle
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) },
    { SpecializationConstant::Type::bool_type, m_isFirstLayer },
  //  { SpecializationConstant::Type::float_type, m_dropoutRate }
  };

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string source = loadFile("./shaders/dense_train_forward.glsl");

  Size3 workSize{ static_cast<uint32_t>(m_size), 1, 1 };

  m_trainForwardShader = m_gpu.compileShader(source, buffers, constants, workSize, includesDir);
}

void DenseLayer::createBackpropDeltaShader(GpuBufferHandle statusBuffer,
  GpuBufferHandle inputBuffer, const Layer* nextLayer) {

  GpuBufferBindings buffers{
    statusBuffer,
    inputBuffer,
    m_bufferB.handle,
    m_bufferW.handle,
    m_bufferZ.handle,
    m_bufferA.handle,
    m_bufferD.handle,
    nextLayer->weightsBuffer(),
    nextLayer->deltaBuffer(),
    m_bufferDeltaB.handle,
    m_bufferDeltaW.handle
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(nextLayer->size()) },
    { SpecializationConstant::Type::bool_type, m_isFirstLayer },
  };

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string source = loadFile("./shaders/dense_backprop_delta.glsl");

  Size3 workSize{ static_cast<uint32_t>(m_size), 1, 1 };

  m_backpropDeltaShader = m_gpu.compileShader(source, buffers, constants, workSize, includesDir);
}

void DenseLayer::createBackpropInputDeltaShader() {
  GpuBufferBindings buffers{
    m_bufferW.handle,
    m_bufferD.handle,
    m_bufferInputDelta.handle
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_size) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) }
  };

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string source = loadFile("./shaders/dense_backprop_input_delta.glsl");

  Size3 workSize{ static_cast<uint32_t>(m_inputSize), 1, 1 };

  m_backpropInputDeltaShader = m_gpu.compileShader(source, buffers, constants, workSize,
    includesDir);
}

void DenseLayer::createUpdateParamsShader(GpuBufferHandle statusBuffer) {
  GpuBufferBindings buffers{
    statusBuffer,
    m_bufferB.handle,
    m_bufferW.handle,
    m_bufferDeltaB.handle,
    m_bufferDeltaW.handle
  };

  SpecializationConstants constants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) },
    { SpecializationConstant::Type::float_type, m_learnRate },
    { SpecializationConstant::Type::float_type, m_learnRateDecay },
  };

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string source = loadFile("./shaders/dense_update_params.glsl");

  Size3 workSize{ static_cast<uint32_t>(m_size), 1, 1 };

  m_updateParamsShader = m_gpu.compileShader(source, buffers, constants, workSize, includesDir);
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
