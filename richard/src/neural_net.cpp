#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include "neural_net.hpp"
#include "util.hpp"
#include "exception.hpp"
#include "labelled_data_set.hpp"

namespace {

using ActivationFn = std::function<double(double)>;
using CostDerivativesFn = std::function<Vector(const Vector&, const Vector&)>;

const ActivationFn sigmoid = [](double x) -> double {
  return 1.0 / (1.0 + exp(-x));
};

const ActivationFn sigmoidPrime = [](double x) -> double {
  double sigX = sigmoid(x);
  return sigX * (1.0 - sigX);
};

const ActivationFn relu = [](double x) -> double {
  return x < 0.0 ? 0.0 : x;
};

const NeuralNet::CostFn quadradicCost = [](const Vector& actual, const Vector& expected) {
  ASSERT(actual.size() == expected.size());

  return (expected - actual).squareMagnitude() / 2.0;
};

// Partial derivatives of quadraticCost with respect to the activations
const CostDerivativesFn quadraticCostDerivatives = [](const Vector& actual,
                                                      const Vector& expected) -> Vector {
  ASSERT(actual.size() == expected.size());

  return actual - expected;
};

enum class LayerType {
  DENSE,
  CONVOLUTIONAL,
  MAX_POOLING,
  OUTPUT
};

class Layer {
  public:
    virtual LayerType type() const = 0;
    virtual std::array<size_t, 2> outputSize() const = 0;
    virtual const Vector& activations() const = 0;
    virtual const Vector& delta() const = 0;
    virtual void trainForward(const Vector& inputs) = 0;
    virtual Vector evalForward(const Vector& inputs) const = 0;
    virtual void updateDelta(const Vector& layerInputs, const Layer& nextLayer) = 0;
    virtual nlohmann::json getConfig() const = 0;
    virtual void writeToStream(std::ostream& fout) const = 0;
    virtual const Matrix& W() const = 0;

    virtual ~Layer() {}
};

class OutputLayer : public Layer {
  public:
    OutputLayer(size_t numNeurons, size_t inputSize, double learnRate);
    OutputLayer(std::istream& fin, size_t numNeurons, size_t inputSize, double learnRate);

    LayerType type() const override { return LayerType::OUTPUT; }
    std::array<size_t, 2> outputSize() const override;
    const Vector& activations() const override;
    const Vector& delta() const override;
    void trainForward(const Vector& inputs) override;
    Vector evalForward(const Vector& inputs) const override;
    void updateDelta(const Vector&, const Layer&) override { assert(false); }
    void updateDelta(const Vector& layerInputs, const Vector& y);
    nlohmann::json getConfig() const override;
    void writeToStream(std::ostream& fout) const override;
    const Matrix& W() const override;

  private:
    Matrix m_W;
    Vector m_B;
    Vector m_Z;
    Vector m_A;
    Vector m_delta;
    double m_learnRate;
};

class DenseLayer : public Layer {
  public:
    DenseLayer(const nlohmann::json& obj, std::istream& fin, size_t inputSize, double learnRate,
      double dropoutRate);
    DenseLayer(const nlohmann::json& obj, size_t inputSize, double learnRate, double dropoutRate);

    LayerType type() const override { return LayerType::DENSE; }
    std::array<size_t, 2> outputSize() const override;
    const Vector& activations() const override;
    const Vector& delta() const override;
    void trainForward(const Vector& inputs) override;
    Vector evalForward(const Vector& inputs) const override;
    void updateDelta(const Vector& layerInputs, const Layer& nextLayer) override;
    nlohmann::json getConfig() const override;
    void writeToStream(std::ostream& fout) const override;
    const Matrix& W() const override;

  private:
    Matrix m_W;
    Vector m_B;
    Vector m_Z;
    Vector m_A;
    Vector m_delta;
    double m_learnRate;
    double m_dropoutRate;
};

class ConvolutionalLayer : public Layer {
  public:
    ConvolutionalLayer(const nlohmann::json& obj, size_t inputW, size_t inputH, double learnRate);
    ConvolutionalLayer(const nlohmann::json& obj, std::istream& fin, size_t inputW, size_t inputH,
      double learnRate);

    LayerType type() const override { return LayerType::CONVOLUTIONAL; }
    std::array<size_t, 2> outputSize() const override;
    const Vector& activations() const override;
    const Vector& delta() const override;
    void trainForward(const Vector& inputs) override;
    Vector evalForward(const Vector& inputs) const override;
    void updateDelta(const Vector& layerInputs, const Layer& nextLayer) override;
    nlohmann::json getConfig() const override;
    void writeToStream(std::ostream& fout) const override;
    const Matrix& W() const override;

  private:
    Matrix m_W;
    double m_b;
    Vector m_Z;
    Vector m_A;
    Vector m_delta;
    size_t m_inputW;
    size_t m_inputH;
    double m_learnRate;
};

class MaxPoolingLayer : public Layer {
  public:
    MaxPoolingLayer(const nlohmann::json& obj, size_t inputW, size_t inputH);

    LayerType type() const override { return LayerType::MAX_POOLING; }
    std::array<size_t, 2> outputSize() const override;
    const Vector& activations() const override;
    const Vector& delta() const override;
    void trainForward(const Vector& inputs) override;
    Vector evalForward(const Vector& inputs) const override;
    void updateDelta(const Vector& layerInputs, const Layer& nextLayer) override;
    nlohmann::json getConfig() const override;
    void writeToStream(std::ostream& fout) const override {}
    const Matrix& W() const override;

  private:
    Vector m_Z;
    Vector m_delta;
    size_t m_regionW;
    size_t m_regionH;
    size_t m_inputW;
    size_t m_inputH;
    Vector m_mask;
};

struct Hyperparams {
  Hyperparams();
  explicit Hyperparams(const nlohmann::json& obj);

  std::array<size_t, 2> numInputs;
  size_t numOutputs;
  size_t epochs;
  double learnRate;
  double learnRateDecay;
  size_t maxBatchSize;
  double dropoutRate;

  nlohmann::json toJson() const;
};

class NeuralNetImpl : public NeuralNet {
  public:
    using CostFn = std::function<double(const Vector&, const Vector&)>;

    explicit NeuralNetImpl(const nlohmann::json& config);
    explicit NeuralNetImpl(std::istream& s);

    CostFn costFn() const override;
    std::array<size_t, 2> inputSize() const override;
    void writeToStream(std::ostream& s) const override;
    void train(LabelledDataSet& data) override;
    Vector evaluate(const Vector& inputs) const override;

  private:
    double feedForward(const Vector& x, const Vector& y, double dropoutRate);
    nlohmann::json getConfig() const;
    OutputLayer& outputLayer();
    std::unique_ptr<Layer> constructLayer(const nlohmann::json& obj, std::istream& fin,
      const std::array<size_t, 2>& prevLayerSize);
    std::unique_ptr<Layer> constructLayer(const nlohmann::json& obj,
      const std::array<size_t, 2>& prevLayerSize);

    bool m_isTrained;
    Hyperparams m_params;
    std::vector<std::unique_ptr<Layer>> m_layers;
};

OutputLayer::OutputLayer(size_t numNeurons, size_t inputSize, double learnRate)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_learnRate(learnRate) {

  m_B = Vector(numNeurons);
  m_B.randomize(1.0);

  m_W = Matrix(inputSize, numNeurons);
  m_W.randomize(1.0);
}

OutputLayer::OutputLayer(std::istream& fin, size_t numNeurons, size_t inputSize, double learnRate)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_learnRate(learnRate) {

  m_B = Vector(numNeurons);
  fin.read(reinterpret_cast<char*>(m_B.data()), numNeurons * sizeof(double));

  m_W = Matrix(inputSize, numNeurons);
  fin.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

void OutputLayer::writeToStream(std::ostream& fout) const {
  fout.write(reinterpret_cast<const char*>(m_B.data()), m_B.size() * sizeof(double));
  fout.write(reinterpret_cast<const char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

const Vector& OutputLayer::activations() const {
  return m_A;
}

const Vector& OutputLayer::delta() const {
  return m_delta;
}

const Matrix& OutputLayer::W() const {
  return m_W;
}

nlohmann::json OutputLayer::getConfig() const {
  return nlohmann::json();
}

Vector OutputLayer::evalForward(const Vector& x) const {
  return (m_W * x + m_B).transform(sigmoid);
}

std::array<size_t, 2> OutputLayer::outputSize() const {
  return { m_B.size(), 1 };
}

void OutputLayer::trainForward(const Vector& inputs) {
  m_Z = m_W * inputs + m_B;
  m_A = m_Z.transform(sigmoid);
}

void OutputLayer::updateDelta(const Vector& layerInputs, const Vector& y) {
  Vector deltaC = quadraticCostDerivatives(m_A, y);
  m_delta = m_Z.transform(sigmoidPrime).hadamard(deltaC);

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      double dw = layerInputs[k] * m_delta[j] * m_learnRate;
      m_W.set(k, j, m_W.at(k, j) - dw);
    }
  }

  m_B = m_B - m_delta * m_learnRate;
}

DenseLayer::DenseLayer(const nlohmann::json& obj, std::istream& fin, size_t inputSize,
  double learnRate, double dropoutRate)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_learnRate(learnRate)
  , m_dropoutRate(dropoutRate) {

  size_t numNeurons = getOrThrow(obj, "size").get<size_t>();

  m_B = Vector(numNeurons);
  fin.read(reinterpret_cast<char*>(m_B.data()), numNeurons * sizeof(double));

  m_W = Matrix(inputSize, numNeurons);
  fin.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

DenseLayer::DenseLayer(const nlohmann::json& obj, size_t inputSize, double learnRate,
  double dropoutRate)
  : m_W(1, 1)
  , m_B(1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_learnRate(learnRate)
  , m_dropoutRate(dropoutRate) {

  size_t numNeurons = getOrThrow(obj, "size").get<size_t>();

  m_B = Vector(numNeurons);
  m_B.randomize(1.0);

  m_W = Matrix(inputSize, numNeurons);
  m_W.randomize(1.0);
}

void DenseLayer::writeToStream(std::ostream& fout) const {
  fout.write(reinterpret_cast<const char*>(m_B.data()), m_B.size() * sizeof(double));
  fout.write(reinterpret_cast<const char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

std::array<size_t, 2> DenseLayer::outputSize() const {
  return { m_B.size(), 1 };
}

const Vector& DenseLayer::activations() const {
  return m_A;
}

const Vector& DenseLayer::delta() const {
  return m_delta;
}

const Matrix& DenseLayer::W() const {
  return m_W;
}

nlohmann::json DenseLayer::getConfig() const {
  nlohmann::json config;
  config["type"] = "dense";
  config["size"] = m_B.size();
  return config;
}

Vector DenseLayer::evalForward(const Vector& x) const {
  return (m_W * x + m_B).transform(sigmoid);
}

void DenseLayer::trainForward(const Vector& inputs) {
  auto shouldDrop = [this]() {
    return rand() / (RAND_MAX + 1.0) < m_dropoutRate;
  };

  m_Z = m_W * inputs + m_B;
  m_A = m_Z.transform(sigmoid);

  for (size_t a = 0; a < m_A.size(); ++a) {
    if (shouldDrop()) {
      m_A[a] = 0.0;
    }
  }
}

void DenseLayer::updateDelta(const Vector& layerInputs, const Layer& nextLayer) {
  m_delta = nextLayer.W().transposeMultiply(nextLayer.delta())
                         .hadamard(m_Z.transform(sigmoidPrime));

  for (size_t j = 0; j < m_W.rows(); j++) {
    for (size_t k = 0; k < m_W.cols(); k++) {
      double dw = layerInputs[k] * m_delta[j] * m_learnRate;
      m_W.set(k, j, m_W.at(k, j) - dw);
    }
  }

  m_B = m_B - m_delta * m_learnRate;
}

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, size_t inputW, size_t inputH,
  double learnRate)
  : m_W(1, 1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_learnRate(learnRate) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();

  m_W = Matrix(kernelSize[0], kernelSize[1]);

  //m_W = Matrix(width * height, 1);

  m_W.randomize(1.0);

  m_b = 0.0; // TODO: Randomize?
}

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, std::istream& fin, size_t inputW,
  size_t inputH, double learnRate)
  : m_W(1, 1)
  , m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_learnRate(learnRate) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();

  m_W = Matrix(kernelSize[0], kernelSize[1]);

  fin.read(reinterpret_cast<char*>(&m_b), sizeof(double));
  fin.read(reinterpret_cast<char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

const Vector& ConvolutionalLayer::activations() const {
  return m_A;
}

const Vector& ConvolutionalLayer::delta() const {
  return m_delta;
}

std::array<size_t, 2> ConvolutionalLayer::outputSize() const {
  return { m_inputW - m_W.cols(), m_inputH - m_W.rows() };
}

void ConvolutionalLayer::trainForward(const Vector& inputs) {
  size_t featureMapW = outputSize()[0];
  size_t featureMapH = outputSize()[1];

  m_Z = Vector(featureMapW * featureMapH);
  m_A = Vector(featureMapW * featureMapH);
  m_delta = Vector(featureMapW * featureMapH);
  for (size_t ymin = 0; ymin < featureMapH; ++ymin) {
    for (size_t xmin = 0; xmin < featureMapW; ++xmin) {
      m_Z[ymin * featureMapW + xmin] = 0.0;
      for (size_t j = 0; j < m_W.rows(); ++j) {
        for (size_t i = 0; i < m_W.cols(); ++i) {
          size_t inputX = xmin + i;
          size_t inputY = ymin + j;
          m_Z[ymin * featureMapW + xmin] += m_W.at(i, j) * inputs[inputY * m_inputW + inputX];
        }
      }
    }
  }

  m_A = m_Z.transform(relu);
}

Vector ConvolutionalLayer::evalForward(const Vector& inputs) const {
  size_t featureMapW = m_inputW - m_W.cols();
  size_t featureMapH = m_inputH - m_W.rows();

  Vector A(featureMapW * featureMapH);
  for (size_t ymin = 0; ymin < featureMapH; ++ymin) {
    for (size_t xmin = 0; xmin < featureMapW; ++xmin) {
      for (size_t j = 0; j < m_W.rows(); ++j) {
        for (size_t i = 0; i < m_W.cols(); ++i) {
          size_t inputX = xmin + i;
          size_t inputY = ymin + j;
          A[ymin * featureMapW + xmin] += relu(m_W.at(i, j) * inputs[inputY * m_inputW + inputX]);
        }
      }
    }
  }

  return A;
}

void ConvolutionalLayer::updateDelta(const Vector& layerInputs, const Layer& nextLayer) {
  TRUE_OR_THROW(nextLayer.type() == LayerType::MAX_POOLING,
    "Expect max pooling after convolutional layer");

  const MaxPoolingLayer& poolingLayer = dynamic_cast<const MaxPoolingLayer&>(nextLayer);
  const Vector& nextLayerDelta = poolingLayer.delta();

  size_t featureMapW = outputSize()[0];
  size_t featureMapH = outputSize()[1];

  double learnRate = m_learnRate / (featureMapW * featureMapH);

  for (size_t j = 0; j < featureMapH; ++j) {
    for (size_t i = 0; i < featureMapW; ++i) {
      size_t idx = j * featureMapW + i;
      m_delta[idx] = nextLayerDelta[idx];
      for (size_t b = 0; b < m_W.rows(); ++b) {
        for (size_t a = 0; a < m_W.cols(); ++a) {
          size_t x = i + a;
          size_t y = j + b;
          double dw = layerInputs[y * m_inputW + x] * m_delta[idx] * learnRate;
          m_W.set(b, a, m_W.at(b, a) - dw);
        }
      }
      m_b = m_b - m_delta[idx] * learnRate;
    }
  }

  //std::cout << "Convolutional delta: \n";
  //std::cout << m_delta;
}

nlohmann::json ConvolutionalLayer::getConfig() const {
  nlohmann::json config;
  config["type"] = "convolutional";
  config["kernelSize"] = std::array<size_t, 2>({ m_W.cols(), m_W.rows() });
  return config;
}

void ConvolutionalLayer::writeToStream(std::ostream& fout) const {
  fout.write(reinterpret_cast<const char*>(&m_b), sizeof(double));
  fout.write(reinterpret_cast<const char*>(m_W.data()), m_W.rows() * m_W.cols() * sizeof(double));
}

const Matrix& ConvolutionalLayer::W() const {
  return m_W;
}

MaxPoolingLayer::MaxPoolingLayer(const nlohmann::json& obj, size_t inputW, size_t inputH)
  : m_Z(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_mask(inputW * inputH) {

  std::array<size_t, 2> regionSize = getOrThrow(obj, "regionSize").get<std::array<size_t, 2>>();
  m_regionW = regionSize[0];
  m_regionH = regionSize[1];

  TRUE_OR_THROW(inputW % m_regionW == 0, "region width does not divide input width");
  TRUE_OR_THROW(inputH % m_regionH == 0, "region height does not divide input height");
}

const Matrix& MaxPoolingLayer::W() const {
  assert(false);
  static Matrix m(1, 1);
  return m;
}

std::array<size_t, 2> MaxPoolingLayer::outputSize() const {
  return { static_cast<size_t>(m_inputW / m_regionW), static_cast<size_t>(m_inputH / m_regionH) };
}

const Vector& MaxPoolingLayer::activations() const {
  return m_Z;
}

const Vector& MaxPoolingLayer::delta() const {
  return m_delta;
}

void MaxPoolingLayer::trainForward(const Vector& inputs) {
  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;
  m_Z = Vector(outputW * outputH);

  //std::cout << "outputW: " << outputW << "\n";
  //std::cout << "outputH: " << outputH << "\n";
  //std::cout << "m_regionW: " << m_regionW << "\n";
  //std::cout << "m_regionH: " << m_regionH << "\n";

  for (size_t y = 0; y < outputH; ++y) {
    for (size_t x = 0; x < outputW; ++x) {
      double largest = std::numeric_limits<double>::min();
      size_t largestInputX = 0;
      size_t largestInputY = 0;
      for (size_t j = 0; j < m_regionH; ++j) {
        for (size_t i = 0; i < m_regionW; ++i) {
          size_t inputX = x * m_regionW + i;
          size_t inputY = y * m_regionH + j;
          double input = inputs[inputY * m_inputW + inputX];
          if (input > largest) {
            largest = input;
            largestInputX = inputX;
            largestInputY = inputY;
          }
          m_mask[inputY * m_inputW + inputX] = 0.0;
        }
      }
      m_mask[largestInputY * m_inputW + largestInputX] = 1.0;
      m_Z[y * outputW + x] = largest;
    }
  }

  //std::cout << "Mask\n";
  //std::cout << m_mask;
}

Vector MaxPoolingLayer::evalForward(const Vector& inputs) const {
  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;
  Vector Z(outputW * outputH);

  for (size_t y = 0; y < outputH; ++y) {
    for (size_t x = 0; x < outputW; ++x) {
      double largest = std::numeric_limits<double>::min();
      for (size_t j = 0; j < m_regionH; ++j) {
        for (size_t i = 0; i < m_regionW; ++i) {
          size_t inputX = x * m_regionW + i;
          size_t inputY = y * m_regionH + j;
          double input = inputs[inputY * m_inputW + inputX];
          if (input > largest) {
            largest = input;
          }
        }
      }
      Z[y * outputW + x] = largest;
    }
  }

  return Z;
}

void MaxPoolingLayer::updateDelta(const Vector& layerInputs, const Layer& nextLayer) {
  m_delta = Vector(m_inputW * m_inputH);

  Vector delta = nextLayer.W().transposeMultiply(nextLayer.delta());

  size_t outputW = m_inputW / m_regionW;
  size_t outputH = m_inputH / m_regionH;

  for (size_t y = 0; y < outputH; ++y) {
    for (size_t x = 0; x < outputW; ++x) {
      for (size_t j = 0; j < m_regionH; ++j) {
        for (size_t i = 0; i < m_regionW; ++i) {
          size_t inputX = x * m_regionW + i;
          size_t inputY = y * m_regionH + j;
          if (m_mask[inputY * m_inputW + inputX] != 0.0) {
            m_delta[inputY * m_inputW + inputX] = delta[y * outputW + x];
          }
          else {
            m_delta[inputY * m_inputW + inputX] = 0.0;
          }
        }
      }
    }
  }

  //std::cout << "Max pooling delta: \n";
  //std::cout << m_delta;
}

nlohmann::json MaxPoolingLayer::getConfig() const {
  nlohmann::json config;
  config["type"] = "maxPooling";
  config["regionSize"] = std::array<size_t, 2>({ m_regionW, m_regionH });
  return config;
}

Hyperparams::Hyperparams()
  : numInputs({784, 1})
  , numOutputs(10)
  , epochs(50)
  , learnRate(0.7)
  , learnRateDecay(1.0)
  , maxBatchSize(1000)
  , dropoutRate(0.5) {}

Hyperparams::Hyperparams(const nlohmann::json& obj) {
  nlohmann::json params = Hyperparams().toJson();
  params.merge_patch(obj);
  numInputs = getOrThrow(params, "numInputs").get<std::array<size_t, 2>>();
  numOutputs = getOrThrow(params, "numOutputs").get<size_t>();
  epochs = getOrThrow(params, "epochs").get<size_t>();
  learnRate = getOrThrow(params, "learnRate").get<double>();
  learnRateDecay = getOrThrow(params, "learnRateDecay").get<double>();
  maxBatchSize = getOrThrow(params, "maxBatchSize").get<size_t>();
  dropoutRate = getOrThrow(params, "dropoutRate").get<double>();
}

nlohmann::json Hyperparams::toJson() const {
  nlohmann::json obj;

  obj["numInputs"] = numInputs;
  obj["numOutputs"] = numOutputs;
  obj["epochs"] = epochs;
  obj["learnRate"] = learnRate;
  obj["learnRateDecay"] = learnRateDecay;
  obj["maxBatchSize"] = maxBatchSize;
  obj["dropoutRate"] = dropoutRate;

  return obj;
}

std::unique_ptr<Layer> NeuralNetImpl::constructLayer(const nlohmann::json& obj, std::istream& fin,
  const std::array<size_t, 2>& prevLayerSize) {

  size_t numInputs = prevLayerSize[0] * prevLayerSize[1];

  std::string type = getOrThrow(obj, "type");
  if (type == "dense") {
    return std::make_unique<DenseLayer>(obj, fin, numInputs, m_params.learnRate,
      m_params.dropoutRate);
  }
  else if (type == "convolutional") {
    return std::make_unique<ConvolutionalLayer>(obj, fin, prevLayerSize[0], prevLayerSize[1],
      m_params.learnRate);
  }
  else if (type == "maxPooling") {
    return std::make_unique<MaxPoolingLayer>(obj, prevLayerSize[0], prevLayerSize[1]);
  }
  else {
    EXCEPTION("Don't know how to construct layer of type '" << type << "'");
  }
}

std::unique_ptr<Layer> NeuralNetImpl::constructLayer(const nlohmann::json& obj,
  const std::array<size_t, 2>& prevLayerSize) {

  size_t numInputs = prevLayerSize[0] * prevLayerSize[1];

  std::string type = getOrThrow(obj, "type");
  if (type == "dense") {
    return std::make_unique<DenseLayer>(obj, numInputs, m_params.learnRate,
      m_params.dropoutRate);
  }
  else if (type == "convolutional") {
    return std::make_unique<ConvolutionalLayer>(obj, prevLayerSize[0], prevLayerSize[1],
      m_params.learnRate);
  }
  else if (type == "maxPooling") {
    return std::make_unique<MaxPoolingLayer>(obj, prevLayerSize[0], prevLayerSize[1]);
  }
  else {
    EXCEPTION("Don't know how to construct layer of type '" << type << "'");
  }
}

NeuralNetImpl::NeuralNetImpl(const nlohmann::json& config)
  : m_isTrained(false)
  , m_params(getOrThrow(config, "hyperparams")) {

  auto prevLayerSize = m_params.numInputs;
  if (config.contains("hiddenLayers")) {
    auto layersJson = config["hiddenLayers"];

    for (auto layerJson : layersJson) {
      m_layers.push_back(constructLayer(layerJson, prevLayerSize));
      prevLayerSize = m_layers.back()->outputSize();
    }
  }

  m_layers.push_back(std::make_unique<OutputLayer>(m_params.numOutputs,
    prevLayerSize[0] * prevLayerSize[1], m_params.learnRate));
}

NeuralNetImpl::NeuralNetImpl(std::istream& fin) : m_isTrained(false) {
  size_t configSize = 0;
  fin.read(reinterpret_cast<char*>(&configSize), sizeof(size_t));

  std::string configString(configSize, '_');
  fin.read(reinterpret_cast<char*>(configString.data()), configSize);
  nlohmann::json config = nlohmann::json::parse(configString);

  nlohmann::json paramsJson = getOrThrow(config, "hyperparams");
  nlohmann::json layersJson = getOrThrow(config, "hiddenLayers");

  m_params = Hyperparams(paramsJson);

  auto prevLayerSize = m_params.numInputs;
  for (auto& layerJson : layersJson) {
    m_layers.push_back(constructLayer(layerJson, fin, prevLayerSize));
    prevLayerSize = m_layers.back()->outputSize();
  }
  m_layers.push_back(std::make_unique<OutputLayer>(fin, m_params.numOutputs,
    prevLayerSize[0] * prevLayerSize[1], m_params.learnRate));

  m_isTrained = true;
}

NeuralNet::CostFn NeuralNetImpl::costFn() const {
  return quadradicCost;
}

nlohmann::json NeuralNetImpl::getConfig() const {
  nlohmann::json config;
  config["hyperparams"] = m_params.toJson();
  std::vector<nlohmann::json> layerJsons;
  for (auto& pLayer : m_layers) {
    layerJsons.push_back(pLayer->getConfig());
  }
  layerJsons.pop_back(); // Omit the output layer
  config["hiddenLayers"] = layerJsons;
  return config;
}

void NeuralNetImpl::writeToStream(std::ostream& fout) const {
  TRUE_OR_THROW(m_isTrained, "Neural net is not trained");

  std::string configString = getConfig().dump();
  size_t configSize = configString.size();
  fout.write(reinterpret_cast<char*>(&configSize), sizeof(size_t));
  fout.write(configString.c_str(), configSize);

  for (const auto& pLayer : m_layers) {
    pLayer->writeToStream(fout);
  }
}

std::array<size_t, 2> NeuralNetImpl::inputSize() const {
  return m_params.numInputs;
}

std::ostream& operator<<(std::ostream& os, LayerType layerType) {
  switch (layerType) {
    case LayerType::DENSE: os << "DENSE"; break;
    case LayerType::CONVOLUTIONAL: os << "CONVOLUTIONAL"; break;
    case LayerType::OUTPUT: os << "OUTPUT"; break;
    case LayerType::MAX_POOLING: os << "MAX_POOLING"; break;
  }
  return os;
}

double NeuralNetImpl::feedForward(const Vector& x, const Vector& y, double dropoutRate) {
  const Vector* A = &x;
  for (auto& layer : m_layers) {
    //std::cout << "Layer type: " << layer->type() << "\n";
    //std::cout << "In: \n";
    //std::cout << *A;
    layer->trainForward(*A);
    //std::cout << "Out: \n";
    A = &layer->activations();
    //std::cout << *A;
  }

  return quadradicCost(*A, y);
}

OutputLayer& NeuralNetImpl::outputLayer() {
  TRUE_OR_THROW(!m_layers.empty(), "No output layer");
  return dynamic_cast<OutputLayer&>(*m_layers.back());
}

void NeuralNetImpl::train(LabelledDataSet& trainingData) {
  double learnRate = m_params.learnRate;

  std::cout << "Epochs: " << m_params.epochs << std::endl;
  std::cout << "Initial learn rate: " << m_params.learnRate << std::endl;
  std::cout << "Learn rate decay: " << m_params.learnRateDecay << std::endl;
  std::cout << "Max batch size: " << m_params.maxBatchSize << std::endl;
  std::cout << "Dropout rate: " << m_params.dropoutRate << std::endl;

  const size_t N = 500; // TODO

  for (size_t epoch = 0; epoch < m_params.epochs; ++epoch) {
    std::cout << "Epoch " << epoch + 1 << "/" << m_params.epochs;
    double cost = 0.0;
    size_t samplesProcessed = 0;

    std::vector<Sample> samples;
    while (trainingData.loadSamples(samples, N) > 0) {
      size_t netInputSize = m_params.numInputs[0] * m_params.numInputs[1];
      TRUE_OR_THROW(samples[0].data.size() == netInputSize,
        "Sample size is " << samples[0].data.size() << ", expected " << netInputSize);

      for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        const Vector& x = sample.data;
        const Vector& y = trainingData.classOutputVector(sample.label);

        cost += feedForward(x, y, m_params.dropoutRate);

        for (int l = static_cast<int>(m_layers.size()) - 1; l >= 0; --l) {
          if (l == static_cast<int>(m_layers.size()) - 1) {
            outputLayer().updateDelta(m_layers[m_layers.size() - 2]->activations(), y);
          }
          else if (l == 0) {
            m_layers[l]->updateDelta(x, *m_layers[l + 1]);
          }
          else {
            m_layers[l]->updateDelta(m_layers[l - 1]->activations(), *m_layers[l + 1]);
          }
        }

        ++samplesProcessed;
        if (samplesProcessed >= m_params.maxBatchSize) {
          break;
        }
      }

      samples.clear();

      if (samplesProcessed >= m_params.maxBatchSize) {
        break;
      }
    }

    learnRate *= m_params.learnRateDecay;

    cost = cost / samplesProcessed;
    std::cout << ", cost = " << cost << std::endl;

    if (std::isnan(cost)) {
      exit(1); // TODO
    }

    trainingData.seekToBeginning();
  }

  m_isTrained = true;
}

Vector NeuralNetImpl::evaluate(const Vector& x) const {
  Vector A = x;
  for (const auto& layer : m_layers) {
    A = layer->evalForward(A);
  }

  return A;
}

}

const nlohmann::json& NeuralNet::defaultConfig() {
  static nlohmann::json config;
  static bool done = false;

  if (!done) {
    nlohmann::json layer1;
    layer1["type"] = "dense";
    layer1["size"] = 300;
    nlohmann::json layer2;
    layer2["type"] = "dense";
    layer2["size"] = 80;
    std::vector<nlohmann::json> layersJson{layer1, layer2};

    config["hyperparams"] = Hyperparams().toJson();
    config["hiddenLayers"] = layersJson;

    done = true;
  }

  return config;
}

std::unique_ptr<NeuralNet> createNeuralNet(const nlohmann::json& config) {
  return std::make_unique<NeuralNetImpl>(config);
}

std::unique_ptr<NeuralNet> createNeuralNet(std::istream& fin) {
  return std::make_unique<NeuralNetImpl>(fin);
}
