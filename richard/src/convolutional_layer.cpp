#include <iostream> // TODO
#include <random>
#include "convolutional_layer.hpp"
#include "max_pooling_layer.hpp"
#include "exception.hpp"
#include "util.hpp"

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, size_t inputW, size_t inputH,
  size_t inputDepth)
  : m_Z(1, 1, 1)
  , m_A(1, 1, 1)
  , m_delta(1, 1, 1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_inputDepth(inputDepth) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();
  size_t depth = getOrThrow(obj, "depth").get<size_t>();

  for (size_t i = 0; i < depth; ++i) {
    Filter filter;

    filter.K = Kernel(kernelSize[0], kernelSize[1], inputDepth);
    filter.K.randomize(0.1);

    filter.b = 0.0;

    m_filters.push_back(filter);
  }

  auto sz = outputSize();

  m_Z = Array3(sz[0], sz[1], sz[2]);
  m_A = Array3(sz[0], sz[1], sz[2]);
  m_delta = Array3(sz[0], sz[1], sz[2]);
}

ConvolutionalLayer::ConvolutionalLayer(const nlohmann::json& obj, std::istream& fin, size_t inputW,
  size_t inputH, size_t inputDepth)
  : m_Z(1, 1, 1)
  , m_A(1, 1, 1)
  , m_delta(1, 1, 1)
  , m_inputW(inputW)
  , m_inputH(inputH)
  , m_inputDepth(inputDepth) {

  std::array<size_t, 2> kernelSize = getOrThrow(obj, "kernelSize").get<std::array<size_t, 2>>();
  m_learnRate = getOrThrow(obj, "learnRate").get<double>();
  m_learnRateDecay = getOrThrow(obj, "learnRateDecay").get<double>();
  size_t depth = getOrThrow(obj, "depth").get<size_t>();

  for (size_t i = 0; i < depth; ++i) {
    Filter filter;

    filter.K = Kernel(kernelSize[0], kernelSize[1], inputDepth);

    fin.read(reinterpret_cast<char*>(&filter.b), sizeof(double));
    fin.read(reinterpret_cast<char*>(filter.K.data()),
      filter.K.W() * filter.K.H() * filter.K.D() * sizeof(double));

    m_filters.push_back(filter);
  }

  auto sz = outputSize();

  m_Z = Array3(sz[0], sz[1], sz[2]);
  m_A = Array3(sz[0], sz[1], sz[2]);
  m_delta = Array3(sz[0], sz[1], sz[2]);
}

const DataArray& ConvolutionalLayer::activations() const {
  return m_A.storage();
}

const DataArray& ConvolutionalLayer::delta() const {
  return m_delta.storage();
}

std::array<size_t, 3> ConvolutionalLayer::outputSize() const {
  ASSERT(!m_filters.empty());
  return {
    m_inputW - m_filters[0].K.W() + 1,
    m_inputH - m_filters[0].K.H() + 1,
    m_filters.size()
  };
}

size_t ConvolutionalLayer::numOutputs() const {
  auto os = outputSize();
  return os[0] * os[1] * os[2];
}

void ConvolutionalLayer::forwardPass(const Array3& inputs, Array3& Z) const {/*
  size_t depth = m_filters.size();

  for (size_t slice = 0; slice < depth; ++slice) {
    Array2Ptr featureMap = Z.submatrix(slice);

    const Kernel& K = m_filters[slice].K;
    double b = m_filters[slice].b;

    K.convolve(inputs, *featureMap);
    (*featureMap) += b;
  }*/
}

void ConvolutionalLayer::trainForward(const DataArray& inputs) {/*
  TRUE_OR_THROW(inputs.dimensions() == 3, "Expected 3 dimensional input to convolutional layer");
  const Array3& image = dynamic_cast<const Array3&>(inputs);

  forwardPass(image, m_Z);

  m_A = m_Z;
  m_A.transformInPlace(relu);*/
}

DataArray ConvolutionalLayer::evalForward(const DataArray& inputs) const {/*
  TRUE_OR_THROW(inputs.dimensions() == 3, "Expected 3 dimensional input to convolutional layer");
  const Array3& image = dynamic_cast<const Array3&>(inputs);

  auto sz = outputSize();
  Array3Ptr Z = std::make_unique<Array3>(sz[0], sz[1], sz[2]);

  forwardPass(image, *Z);

  Z->transformInPlace(relu);

  return Z;*/
}

// TODO
void ConvolutionalLayer::updateDelta(const DataArray& inputs, const Layer& nextLayer,
  size_t epoch) {/*

  TRUE_OR_THROW(nextLayer.type() == LayerType::MAX_POOLING,
    "Expect max pooling after convolutional layer");

  const Vector& nextLayerDelta = nextLayer.delta();

  size_t featureMapW = outputSize()[0];
  size_t featureMapH = outputSize()[1];
  size_t depth = m_filters.size();
  size_t sliceSize = featureMapW * featureMapH;

  double learnRate = m_learnRate * pow(m_learnRateDecay, epoch) /
    (m_inputDepth * depth * featureMapW * featureMapH);

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
  //std::cout << m_delta;*/
}

nlohmann::json ConvolutionalLayer::getConfig() const {
  ASSERT(!m_filters.empty());

  nlohmann::json config;
  config["type"] = "convolutional";
  config["kernelSize"] = std::array<size_t, 2>({ m_filters[0].K.W(), m_filters[0].K.H() });
  config["depth"] = m_filters.size();
  config["learnRate"] = m_learnRate;
  config["learnRateDecay"] = m_learnRateDecay;
  return config;
}

void ConvolutionalLayer::writeToStream(std::ostream& fout) const {/*
  for (const LayerParams& slice : m_slices) {
    fout.write(reinterpret_cast<const char*>(&slice.b), sizeof(double));
    fout.write(reinterpret_cast<const char*>(slice.W.data()),
      slice.W.rows() * slice.W.cols() * sizeof(double));
  }*/
}

const Matrix& ConvolutionalLayer::W() const {
  assert(false);
  static Matrix m(1, 1);
  return m;
}

size_t ConvolutionalLayer::depth() const {
  return m_filters.size();
}

const std::vector<ConvolutionalLayer::Filter>& ConvolutionalLayer::filters() const {
  return m_filters;
}

std::array<size_t, 2> ConvolutionalLayer::kernelSize() const {
  ASSERT(m_filters.size() > 0);
  return { m_filters[0].K.W(), m_filters[0].K.H() };
}

/*
// For testing
void ConvolutionalLayer::setWeights(const std::vector<Matrix>& weights) {
  ASSERT(weights.size() == m_slices.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    m_slices[i].W = weights[i];
  }
}

// For testing
void ConvolutionalLayer::setBiases(const std::vector<double>& biases) {
  ASSERT(biases.size() == m_slices.size());
  for (size_t i = 0; i < biases.size(); ++i) {
    m_slices[i].b = biases[i];
  }
}
*/
