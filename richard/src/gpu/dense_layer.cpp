#include "gpu/dense_layer.hpp"
#include "utils.hpp"

namespace richard {
namespace gpu {

DenseLayer::DenseLayer(Gpu& gpu, const nlohmann::json& obj, size_t inputSize, bool isFirstLayer)
  : m_gpu(gpu) {

  initialize(obj, inputSize, isFirstLayer);
  m_W.randomize(0.1);
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
}

void DenseLayer::allocateGpuBuffers() {
  GpuBufferFlags paramBuffersFlags = GpuBufferFlags::large
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;

  m_bufferB = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), paramBuffersFlags);
  m_bufferW = m_gpu.allocateBuffer(m_inputSize * m_size * sizeof(netfloat_t), paramBuffersFlags);
  m_bufferZ = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), GpuBufferFlags::large);
  m_bufferA = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), GpuBufferFlags::large);
  m_bufferD = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), GpuBufferFlags::large);
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

  GpuBufferBindings evalForwardBuffers{
    inputBuffer,
    m_bufferB.handle,
    m_bufferW.handle,
    m_bufferA.handle
  };

  GpuBufferBindings trainForwardBuffers{
    statusBuffer,
    inputBuffer,
    m_bufferB.handle,
    m_bufferW.handle,
    m_bufferZ.handle,
    m_bufferA.handle
  };

  GpuBufferBindings backpropBuffers{
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

  GpuBufferBindings updateParamsBuffers{
    statusBuffer,
    m_bufferB.handle,
    m_bufferW.handle,
    m_bufferDeltaB.handle,
    m_bufferDeltaW.handle
  };

  const size_t maxWorkgroupSize = 64; // TODO

  Size3 workgroupSize{ static_cast<uint32_t>(std::min(m_size, maxWorkgroupSize)), 1, 1 };
  Size3 numWorkgroups{ m_size / workgroupSize[0], 1, 1 };

  ASSERT_MSG(workgroupSize[0] * numWorkgroups[0] == m_size,
    "Layer size " << m_size << " is not divisible by workgroup size " << workgroupSize[0]);

  SpecializationConstants evalForwardConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) }
  };

  SpecializationConstants trainForwardConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) },
    { SpecializationConstant::Type::bool_type, m_isFirstLayer },
  //  { SpecializationConstant::Type::float_type, m_dropoutRate }
  };

  SpecializationConstants backpropConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) },
    { SpecializationConstant::Type::bool_type, m_isFirstLayer },
  };

  SpecializationConstants updateParamsConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) },
    { SpecializationConstant::Type::float_type, m_learnRate },
    { SpecializationConstant::Type::float_type, m_learnRateDecay },
  };

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string evalForwardSrc = loadFile("./shaders/dense_eval_forward.glsl");
  const std::string trainForwardSrc = loadFile("./shaders/dense_train_forward.glsl");
  const std::string backpropSrc = loadFile("./shaders/dense_backprop.glsl");
  const std::string updateParamsSrc = loadFile("./shaders/dense_update_params.glsl");

  m_evalForwardShader = m_gpu.compileShader(evalForwardSrc, evalForwardBuffers,
    evalForwardConstants, workgroupSize, numWorkgroups, includesDir);
  m_trainForwardShader = m_gpu.compileShader(trainForwardSrc, trainForwardBuffers,
    trainForwardConstants, workgroupSize, numWorkgroups, includesDir);
  m_backpropShader = m_gpu.compileShader(backpropSrc, backpropBuffers, backpropConstants,
    workgroupSize, numWorkgroups, includesDir);
  m_updateParamsShader = m_gpu.compileShader(updateParamsSrc, updateParamsBuffers,
    updateParamsConstants, workgroupSize, numWorkgroups, includesDir);
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
  m_gpu.queueShader(m_backpropShader);
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
  EXCEPTION("Dense layer does not expose input delta buffer");
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

GpuBufferHandle DenseLayer::test_activationsBuffer() const {
  return m_bufferA.handle;
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
