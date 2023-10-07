#include <iostream> // TODO
#include <random>
#include "convolutional_layer.hpp"
#include "max_pooling_layer.hpp"
#include "exception.hpp"

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, size_t inputW, size_t inputH)
  : m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();
  size_t depth = getOrThrow(obj, "depth").get<size_t>();

  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (size_t i = 0; i < depth; ++i) {
    SliceParams slice;

    slice.W = Matrix(kernelSize[0], kernelSize[1]);
    slice.W.randomize(1.0);

    slice.b = dist(gen);

    m_slices.push_back(slice);
  }
}

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, std::istream& fin, size_t inputW,
  size_t inputH)
  : m_Z(1)
  , m_A(1)
  , m_delta(1)
  , m_inputW(inputW)
  , m_inputH(inputH) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();
  size_t depth = getOrThrow(obj, "depth").get<size_t>();

  for (size_t i = 0; i < depth; ++i) {
    SliceParams slice;

    slice.W = Matrix(kernelSize[0], kernelSize[1]);

    fin.read(reinterpret_cast<char*>(&slice.b), sizeof(double));
    fin.read(reinterpret_cast<char*>(slice.W.data()),
      slice.W.rows() * slice.W.cols() * sizeof(double));

    m_slices.push_back(slice);
  }
}

const Vector& ConvolutionalLayer::activations() const {
  return m_A;
}

const Vector& ConvolutionalLayer::delta() const {
  return m_delta;
}

std::array<size_t, 3> ConvolutionalLayer::outputSize() const {
  ASSERT(!m_slices.empty());
  return { m_inputW - m_slices[0].W.cols(), m_inputH - m_slices[0].W.rows(), m_slices.size() };
}

size_t ConvolutionalLayer::sliceSize() const {
  size_t featureMapW = outputSize()[0];
  size_t featureMapH = outputSize()[1];

  return featureMapW * featureMapH;
}

void ConvolutionalLayer::forwardPass(const Vector& inputs, Vector& Z) const {
  size_t featureMapW = outputSize()[0];
  size_t featureMapH = outputSize()[1];

  size_t sliceSize = featureMapW * featureMapH;

  for (size_t slice = 0; slice < m_slices.size(); ++slice) {
    size_t sliceOffset = slice * sliceSize;
    const Matrix& W = m_slices[slice].W;
    double b = m_slices[slice].b;

    for (size_t ymin = 0; ymin < featureMapH; ++ymin) {
      for (size_t xmin = 0; xmin < featureMapW; ++xmin) {
        Z[sliceOffset + ymin * featureMapW + xmin] = b;

        for (size_t j = 0; j < W.rows(); ++j) {
          for (size_t i = 0; i < W.cols(); ++i) {
            size_t inputX = xmin + i;
            size_t inputY = ymin + j;

            Z[sliceOffset + ymin * featureMapW + xmin] +=
              W.at(i, j) * inputs[inputY * m_inputW + inputX];
          }
        }
      }
    }
  }
}

void ConvolutionalLayer::trainForward(const Vector& inputs) {
  size_t sz = sliceSize();

  m_Z = Vector(sz * m_slices.size());
  m_A = Vector(sz * m_slices.size());
  m_delta = Vector(sz * m_slices.size());

  forwardPass(inputs, m_Z);

  m_A = m_Z.transform(relu);
}

Vector ConvolutionalLayer::evalForward(const Vector& inputs) const {
  size_t sz = sliceSize();

  Vector Z(sz * m_slices.size());

  forwardPass(inputs, Z);

  return Z.transform(relu);
}

void ConvolutionalLayer::updateDelta(const Vector& layerInputs, const Layer& nextLayer,
  size_t epoch) {
  
  TRUE_OR_THROW(nextLayer.type() == LayerType::MAX_POOLING,
    "Expect max pooling after convolutional layer");

  const MaxPoolingLayer& poolingLayer = dynamic_cast<const MaxPoolingLayer&>(nextLayer);
  const Vector& nextLayerDelta = poolingLayer.delta();

  size_t featureMapW = outputSize()[0];
  size_t featureMapH = outputSize()[1];
  size_t sliceSize = featureMapW * featureMapH;

  double learnRate = m_learnRate * pow(m_learnRateDecay, epoch) / (featureMapW * featureMapH);

  for (size_t slice = 0; slice < m_slices.size(); ++slice) {
    size_t sliceOffset = slice * sliceSize;
    Matrix& W = m_slices[slice].W;
    double& b = m_slices[slice].b;

    for (size_t ymin = 0; ymin < featureMapH; ++ymin) {
      for (size_t xmin = 0; xmin < featureMapW; ++xmin) {
        size_t idx = sliceOffset + ymin * featureMapW + xmin;

        m_delta[idx] = nextLayerDelta[idx];

        for (size_t j = 0; j < W.rows(); ++j) {
          for (size_t i = 0; i < W.cols(); ++i) {
            size_t inputX = xmin + i;
            size_t inputY = ymin + j;
            double dw = layerInputs[inputY * m_inputW + inputX] * m_delta[idx] * learnRate;

            W.set(j, i, W.at(j, i) - dw);
            b = b - m_delta[idx] * learnRate;
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
  for (const SliceParams& slice : m_slices) {
    fout.write(reinterpret_cast<const char*>(&slice.b), sizeof(double));
    fout.write(reinterpret_cast<const char*>(slice.W.data()),
      slice.W.rows() * slice.W.cols() * sizeof(double));
  }
}

const Matrix& ConvolutionalLayer::W() const {
  assert(false);
  return m_slices[0].W;
}
