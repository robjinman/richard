#include "gpu/output_layer.hpp"
#include "util.hpp"

namespace richard {
namespace gpu {

OutputLayer::OutputLayer(Gpu& gpu, const nlohmann::json& obj, std::istream& stream,
  size_t inputSize, size_t miniBatchSize)
  : m_gpu(gpu)
  , m_miniBatchSize(miniBatchSize)
  , m_inputSize(inputSize) {

  m_size = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<netfloat_t>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<netfloat_t>();

  m_B = Vector(m_size);
  stream.read(reinterpret_cast<char*>(m_B.data()), m_size * sizeof(netfloat_t));

  m_W = Matrix(m_inputSize, m_size);
  stream.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

OutputLayer::OutputLayer(Gpu& gpu, const nlohmann::json& obj, size_t inputSize,
  size_t miniBatchSize)
  : m_gpu(gpu)
  , m_miniBatchSize(miniBatchSize)
  , m_inputSize(inputSize) {

  m_size = getOrThrow(obj, "size").get<size_t>();
  m_learnRate = getOrThrow(obj, "learnRate").get<netfloat_t>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<netfloat_t>();

  m_B = Vector(m_size);
  m_B.randomize(0.1);
  m_W = Matrix(m_inputSize, m_size);
  m_W.randomize(0.1);
}

void OutputLayer::allocateGpuResources(GpuBufferHandle inputBuffer, GpuBufferHandle statusBuffer,
  const Layer*, GpuBufferHandle sampleYBuffer) {

  GpuBufferFlags paramBuffersFlags = GpuBufferFlags::large
                                   | GpuBufferFlags::hostReadAccess
                                   | GpuBufferFlags::hostWriteAccess;

  m_bufferB = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), paramBuffersFlags);
  m_bufferW = m_gpu.allocateBuffer(m_inputSize * m_size * sizeof(netfloat_t), paramBuffersFlags);
  m_bufferZ = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), GpuBufferFlags::large);
  m_bufferA = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), GpuBufferFlags::large);
  m_bufferD = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), GpuBufferFlags::large);
  m_bufferDeltaB = m_gpu.allocateBuffer(m_size * sizeof(netfloat_t), GpuBufferFlags::large);
  m_bufferDeltaW = m_gpu.allocateBuffer(m_inputSize * m_size * sizeof(netfloat_t),
    GpuBufferFlags::large);

  m_gpu.submitBufferData(m_bufferB.handle, m_B.data());
  m_gpu.submitBufferData(m_bufferW.handle, m_W.data());

  GpuBufferBindings evalForwardBuffers{
    inputBuffer,
    m_bufferB.handle,
    m_bufferW.handle,
    m_bufferA.handle
  };

  GpuBufferBindings trainForwardBuffers{
    statusBuffer,
    inputBuffer,
    sampleYBuffer,
    m_bufferB.handle,
    m_bufferW.handle,
    m_bufferZ.handle,
    m_bufferA.handle
  };

  GpuBufferBindings backpropBuffers{
    statusBuffer,
    inputBuffer,
    sampleYBuffer,
    m_bufferB.handle,
    m_bufferW.handle,
    m_bufferZ.handle,
    m_bufferA.handle,
    m_bufferD.handle,
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

  Size3 workgroupSize{ static_cast<uint32_t>(m_size), 1, 1 };

  SpecializationConstants evalForwardConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) }
  };

  SpecializationConstants trainForwardConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_miniBatchSize) }
  };

  SpecializationConstants backpropConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) },
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_miniBatchSize) }
  };

  SpecializationConstants updateParamsConstants{
    { SpecializationConstant::Type::uint_type, static_cast<uint32_t>(m_inputSize) },
    { SpecializationConstant::Type::float_type, m_learnRate },
    { SpecializationConstant::Type::float_type, m_learnRateDecay },
  };

  // TODO: Remove hard-coded paths
  const std::string includesDir = "./shaders";
  const std::string evalForwardSrc = loadFile("./shaders/output_eval_forward.glsl");
  const std::string trainForwardSrc = loadFile("./shaders/output_train_forward.glsl");
  const std::string backpropSrc = loadFile("./shaders/output_backprop.glsl");
  const std::string updateParamsSrc = loadFile("./shaders/output_update_params.glsl");

  m_evalForwardShader = m_gpu.compileShader(evalForwardSrc, evalForwardBuffers,
    evalForwardConstants, workgroupSize, includesDir);
  m_trainForwardShader = m_gpu.compileShader(trainForwardSrc, trainForwardBuffers,
    trainForwardConstants, workgroupSize, includesDir);
  m_backpropShader = m_gpu.compileShader(backpropSrc, backpropBuffers, backpropConstants,
    workgroupSize, includesDir);
  m_updateParamsShader = m_gpu.compileShader(updateParamsSrc, updateParamsBuffers,
    updateParamsConstants, workgroupSize, includesDir);
}

size_t OutputLayer::size() const {
  return m_size;
}

Triple OutputLayer::outputSize() const {
  return { m_size, 1, 1 };
}

void OutputLayer::evalForward() {
  m_gpu.queueShader(m_evalForwardShader);
}

void OutputLayer::trainForward() {
  m_gpu.queueShader(m_trainForwardShader);
}

void OutputLayer::backprop() {
  m_gpu.queueShader(m_backpropShader);
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

void OutputLayer::retrieveBuffers() {
  m_gpu.retrieveBuffer(m_bufferB.handle, m_B.data());
  m_gpu.retrieveBuffer(m_bufferW.handle, m_W.data());
}

void OutputLayer::writeToStream(std::ostream& stream) const {
  stream.write(reinterpret_cast<const char*>(m_B.data()), m_B.size() * sizeof(netfloat_t));
  stream.write(reinterpret_cast<const char*>(m_W.data()),
    m_W.rows() * m_W.cols() * sizeof(netfloat_t));
}

void OutputLayer::setWeights(const Matrix& W) {
  m_W = W;
}

void OutputLayer::setBiases(const Vector& B) {
  m_B = B;
}

GpuBufferHandle OutputLayer::activationsBuffer() const {
  return m_bufferA.handle;
}

}
}
