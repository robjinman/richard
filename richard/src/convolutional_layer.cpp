#include <iostream> // TODO
#include <random>
#include <omp.h>
#include "convolutional_layer.hpp"
#include "max_pooling_layer.hpp"
#include "exception.hpp"

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, size_t inputW, size_t inputH,
  size_t inputDepth)
  : m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_inputDepth(inputDepth) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();
  size_t depth = getOrThrow(obj, "depth").get<size_t>();

  for (size_t i = 0; i < depth; ++i) {
    LayerParams slice;

    slice.W = Matrix(kernelSize[0], kernelSize[1]);
    slice.W.randomize(0.1);

    slice.b = 0.0;

    m_slices.push_back(slice);
  }

  size_t sz = numOutputs();

  m_Z = Vector(sz);
  m_A = Vector(sz);
  m_delta = Vector(sz);
}

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, std::istream& fin, size_t inputW,
  size_t inputH, size_t inputDepth)
  : m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_inputDepth(inputDepth) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();
  size_t depth = getOrThrow(obj, "depth").get<size_t>();

  for (size_t i = 0; i < depth; ++i) {
    LayerParams slice;

    slice.W = Matrix(kernelSize[0], kernelSize[1]);

    fin.read(reinterpret_cast<char*>(&slice.b), sizeof(double));
    fin.read(reinterpret_cast<char*>(slice.W.data()),
      slice.W.rows() * slice.W.cols() * sizeof(double));

    m_slices.push_back(slice);
  }

  size_t sz = numOutputs();

  m_Z = Vector(sz);
  m_A = Vector(sz);
  m_delta = Vector(sz);
}

const Vector& ConvolutionalLayer::activations() const {
  return m_A;
}

const Vector& ConvolutionalLayer::delta() const {
  return m_delta;
}

std::array<size_t, 3> ConvolutionalLayer::outputSize() const {
  ASSERT(!m_slices.empty());
  return {
    m_inputW - m_slices[0].W.cols() + 1,
    m_inputH - m_slices[0].W.rows() + 1,
    m_slices.size() * m_inputDepth
  };
}

size_t ConvolutionalLayer::numOutputs() const {
  auto os = outputSize();
  return os[0] * os[1] * os[2];
}

void ConvolutionalLayer::forwardPass(const Vector& inputs, Vector& Z) const {
  size_t featureMapW = outputSize()[0];
  size_t featureMapH = outputSize()[1];

  size_t depth = m_slices.size();
  size_t sliceSize = featureMapW * featureMapH;

  #pragma omp parallel for
  for (size_t inputSlice = 0; inputSlice < m_inputDepth; ++inputSlice) {
    size_t inputOffset = m_inputW * m_inputH * inputSlice;

    for (size_t slice = 0; slice < m_slices.size(); ++slice) {
      size_t outputOffset = inputSlice * depth * sliceSize + slice * sliceSize;
      const Matrix& W = m_slices[slice].W;
      double b = m_slices[slice].b;

      for (size_t ymin = 0; ymin < featureMapH; ++ymin) {
        for (size_t xmin = 0; xmin < featureMapW; ++xmin) {
          Z[outputOffset + ymin * featureMapW + xmin] = b;

          for (size_t j = 0; j < W.rows(); ++j) {
            for (size_t i = 0; i < W.cols(); ++i) {
              size_t inputX = xmin + i;
              size_t inputY = ymin + j;

              Z[outputOffset + ymin * featureMapW + xmin] +=
                W.at(i, j) * inputs[inputOffset + inputY * m_inputW + inputX];
            }
          }
        }
      }
    }
  }
}

void ConvolutionalLayer::trainForward(const Vector& inputs) {
  forwardPass(inputs, m_Z);

  m_A = m_Z.transform(relu);
}

Vector ConvolutionalLayer::evalForward(const Vector& inputs) const {
  Vector Z(numOutputs());

  forwardPass(inputs, Z);

  return Z.transform(relu);
}

void ConvolutionalLayer::updateDelta(const Vector& layerInputs, const Layer& nextLayer,
  size_t epoch) {

  //TRUE_OR_THROW(nextLayer.type() == LayerType::MAX_POOLING,
  //  "Expect max pooling after convolutional layer");

  const Vector& nextLayerDelta = nextLayer.delta();

  size_t featureMapW = outputSize()[0];
  size_t featureMapH = outputSize()[1];
  size_t depth = m_slices.size();
  size_t sliceSize = featureMapW * featureMapH;

  double learnRate = m_learnRate * pow(m_learnRateDecay, epoch) /
    (m_inputDepth * depth * featureMapW * featureMapH);

  //std::cout << learnRate << "\n";

  // Total number of feature maps is m_inputDepth * depth
  for (size_t inputSlice = 0; inputSlice < m_inputDepth; ++inputSlice) {
    for (size_t slice = 0; slice < depth; ++slice) {
      size_t sliceOffset = inputSlice * depth * sliceSize + slice * sliceSize;
      Matrix& W = m_slices[slice].W;
      double& b = m_slices[slice].b;

      for (size_t ymin = 0; ymin < featureMapH; ++ymin) {
        for (size_t xmin = 0; xmin < featureMapW; ++xmin) {
          size_t idx = sliceOffset + ymin * featureMapW + xmin;

          m_delta[idx] = reluPrime(m_Z[idx]) * nextLayerDelta[idx];

          for (size_t j = 0; j < W.rows(); ++j) {
            for (size_t i = 0; i < W.cols(); ++i) {
              size_t inputX = xmin + i;
              size_t inputY = ymin + j;
              double dw = layerInputs[inputY * m_inputW + inputX] * m_delta[idx] * learnRate;

              W.set(i, j, W.at(i, j) - dw);
              b = b - m_delta[idx] * learnRate;
            }
          }
        }
      }
    }
  }

  //std::cout << "Convolutional delta: \n";
  //std::cout << m_delta;
}

nlohmann::json ConvolutionalLayer::getConfig() const {
  assert(!m_slices.empty());

  nlohmann::json config;
  config["type"] = "convolutional";
  config["kernelSize"] = std::array<size_t, 2>({ m_slices[0].W.cols(), m_slices[0].W.rows() });
  config["depth"] = m_slices.size();
  config["learnRate"] = m_learnRate;
  config["learnRateDecay"] = m_learnRateDecay;
  return config;
}

void ConvolutionalLayer::writeToStream(std::ostream& fout) const {
  for (const LayerParams& slice : m_slices) {
    fout.write(reinterpret_cast<const char*>(&slice.b), sizeof(double));
    fout.write(reinterpret_cast<const char*>(slice.W.data()),
      slice.W.rows() * slice.W.cols() * sizeof(double));
  }
}

const Matrix& ConvolutionalLayer::W() const {
  assert(false);
  return m_slices[0].W;
}

size_t ConvolutionalLayer::depth() const {
  return m_slices.size();
}

const std::vector<LayerParams>& ConvolutionalLayer::params() const {
  return m_slices;
}

std::array<size_t, 2> ConvolutionalLayer::kernelSize() const {
  assert(m_slices.size() > 0);
  return { m_slices[0].W.cols(), m_slices[0].W.rows() };
}

// For testing
void ConvolutionalLayer::setWeights(const std::vector<Matrix>& weights) {
  assert(weights.size() == m_slices.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    m_slices[i].W = weights[i];
  }
}

// For testing
void ConvolutionalLayer::setBiases(const std::vector<double>& biases) {
  assert(biases.size() == m_slices.size());
  for (size_t i = 0; i < biases.size(); ++i) {
    m_slices[i].b = biases[i];
  }
}
