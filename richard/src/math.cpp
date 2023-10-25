#include <iostream>
#include <ostream>
#include <cstring>
#include <random>
#include "math.hpp"
#include "exception.hpp"
#include "util.hpp"

namespace {

bool arraysEqual(const double* A, const double* B, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (A[i] != B[i]) {
      return false;
    }
  }
  
  return true;
}

size_t count(std::initializer_list<std::initializer_list<double>> X) {
  size_t H = X.size();
  ASSERT(H > 0);
  size_t W = X.begin()->size();
  ASSERT(W > 0);

  size_t numElements = 0;

  for (auto row : X) {
    ASSERT(row.size() == W);
    numElements += W;
  }

  return numElements;
}

size_t count(std::initializer_list<std::initializer_list<std::initializer_list<double>>> X) {
  size_t H = X.size();
  ASSERT(H > 0);
  size_t W = X.begin()->size();
  ASSERT(W > 0);
  size_t D = X.begin()->begin()->size();
  ASSERT(D > 0);

  size_t numElements = 0;

  for (auto row : X) {
    ASSERT(row.size() == W);

    for (auto zLine : row) {
      ASSERT(zLine.size() == D);
      numElements += D;
    }
  }

  return numElements;
}

}

DataArray::DataArray()
  : m_data(nullptr)
  , m_size(0) {}

DataArray::DataArray(size_t size)
  : m_data(new double[size])
  , m_size(size) {

  memset(m_data.get(), 0, m_size * sizeof(double));
}

DataArray::DataArray(const DataArray& cpy)
  : m_data(new double[cpy.m_size])
  , m_size(cpy.m_size) {

  memcpy(m_data.get(), cpy.m_data.get(), m_size * sizeof(double));
}

DataArray::DataArray(DataArray&& mv)
  : m_data(std::move(mv.m_data))
  , m_size(mv.m_size) {

  mv.m_size = 0;
}

DataArray& DataArray::operator=(const DataArray& rhs) {
  m_size = rhs.m_size;
  m_data.reset(new double[m_size]);
  memcpy(m_data.get(), rhs.m_data.get(), m_size * sizeof(double));

  return *this;
}

DataArray& DataArray::operator=(DataArray&& rhs) {
  m_size = rhs.m_size;
  m_data = std::move(rhs.m_data);

  rhs.m_size = 0;

  return *this;
}

DataArray DataArray::concat(const DataArray& A, const DataArray& B) {
  DataArray C(A.size() + B.size());

  void* ptr = C.m_data.get();
  memcpy(ptr, A.m_data.get(), A.size() * sizeof(double));
  ptr += A.size() * sizeof(double);
  memcpy(ptr, B.m_data.get(), B.size() * sizeof(double));

  return C;
}

std::ostream& operator<<(std::ostream& os, const DataArray& v) {
  os << "[ ";
  for (size_t i = 0; i < v.size(); ++i) {
    os << v[i] << " ";
  }
  os << "]" << std::endl;

  return os;
}

Vector::Vector(std::initializer_list<double> data)
  : m_storage(data.size())
  , m_data(m_storage.data())
  , m_size(data.size()) {

  size_t i = 0;
  for (double x : data) {
    m_data[i++] = x;
  }
}

Vector::Vector(size_t length)
  : m_storage(length)
  , m_data(m_storage.data())
  , m_size(length) {}

Vector::Vector(double* data, size_t size)
  : m_data(data)
  , m_size(size) {}

Vector::Vector(const DataArray& data)
  : m_storage(data)
  , m_data(m_storage.data())
  , m_size(m_storage.size()) {}

Vector::Vector(DataArray&& data)
  : m_storage(std::move(data))
  , m_data(m_storage.data())
  , m_size(m_storage.size()) {}

Vector::Vector(const Vector& cpy) {
  *this = cpy;
}

Vector::Vector(Vector&& mv) {
  *this = std::move(mv);
}

Vector& Vector::operator=(const Vector& rhs) {
  m_size = rhs.m_size;

  if (rhs.isShallow()) {
    m_storage = DataArray();
    m_data = rhs.m_data;
  }
  else {
    m_storage = DataArray(m_size);
    m_data = m_storage.data();
    memcpy(m_data, rhs.m_data, m_size * sizeof(double));
  }

  return *this;
}

Vector& Vector::operator=(Vector&& rhs) {
  m_size = rhs.m_size;

  if (rhs.isShallow()) {
    m_storage = DataArray();
    m_data = rhs.m_data;
  }
  else {
    m_storage = std::move(rhs.m_storage);
    m_data = m_storage.data();
  }

  rhs.m_data = nullptr;
  rhs.m_size = 0;

  return *this;
}

VectorPtr Vector::clone() const {
  VectorPtr cpy(new Vector(m_size));
  memcpy(cpy->m_data, m_data, m_size * sizeof(double));
  return cpy;
}

bool Vector::operator==(const Vector& rhs) const {
  if (m_size != rhs.m_size) {
    return false;
  }

  return arraysEqual(m_data, rhs.m_data, m_size);
}

double Vector::magnitude() const {
  return sqrt(squareMagnitude());
}

double Vector::squareMagnitude() const {
  double sqSum = 0.0;
  for (size_t i = 0; i < m_size; ++i) {
    double x = m_data[i];
    sqSum += x * x;
  }
  return sqSum;
}

void Vector::zero() {
  memset(m_data, 0, m_size * sizeof(double));
}

void Vector::fill(double x) {
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] = x;
  }
}

void Vector::randomize(double standardDeviation) {
  std::default_random_engine gen;
  std::normal_distribution<double> dist(0.0, standardDeviation);

  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] = dist(gen);
  }
}

void Vector::normalize() {
  double mag = magnitude();
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] = m_data[i] / mag;
  }
}

double Vector::dot(const Vector& rhs) const {
  ASSERT(rhs.m_size == m_size);

  double x = 0.0;
  for (size_t i = 0; i < m_size; ++i) {
    x += m_data[i] * rhs[i];
  }
  return x;
}

Vector Vector::hadamard(const Vector& rhs) const {
  ASSERT(rhs.m_size == m_size);

  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] * rhs[i];
  }
  return v;
}

Vector Vector::operator+(const Vector& rhs) const {
  ASSERT(rhs.m_size == m_size);

  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] + rhs[i];
  }
  return v;
}

Vector Vector::operator-(const Vector& rhs) const {
  ASSERT(rhs.m_size == m_size);

  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] - rhs[i];
  }
  return v;
}

Vector Vector::operator*(double s) const {
  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] * s;
  }
  return v;
}

Vector Vector::operator+(double s) const {
  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] + s;
  }
  return v;
}

Vector Vector::operator-(double s) const {
  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] - s;
  }
  return v;
}

double Vector::sum() const {
  double s = 0.0;

  for (size_t i = 0; i < m_size; ++i) {
    s += m_data[i];
  }

  return s;
}

Vector Vector::computeTransform(const std::function<double(double)>& f) const {
  Vector v(m_size);

  for (size_t i = 0; i < m_size; ++i) {
    v.m_data[i] = f(m_data[i]);
  }

  return v;
}

void Vector::transformInPlace(const std::function<double(double)>& f) {
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] = f(m_data[i]);
  }
}

VectorPtr Vector::createShallow(DataArray& data) {
  return VectorPtr(new Vector(data.data(), data.size()));
}

ConstVectorPtr Vector::createShallow(const DataArray& data) {
  return ConstVectorPtr(new Vector(const_cast<double*>(data.data()), data.size()));
}

std::ostream& operator<<(std::ostream& os, const Vector& v) {
  os << "[ ";
  for (size_t i = 0; i < v.size(); ++i) {
    os << v[i] << " ";
  }
  os << "]" << std::endl;

  return os;
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> data)
  : m_storage(count(data))
  , m_data(m_storage.data())
  , m_rows(data.size())
  , m_cols(data.begin()->size()) {

  size_t r = 0;
  for (auto row : data) {
    size_t c = 0;
    for (double value : row) {
      set(c, r, value);
      ++c;
    }
    ++r;
  }
}

Matrix::Matrix(size_t cols, size_t rows)
  : m_storage(cols * rows)
  , m_data(m_storage.data())
  , m_rows(rows)
  , m_cols(cols) {}

Matrix::Matrix(double* data, size_t cols, size_t rows)
  : m_data(data)
  , m_rows(rows)
  , m_cols(cols) {}

Matrix::Matrix(const DataArray& data, size_t cols, size_t rows)
  : m_storage(data)
  , m_data(m_storage.data())
  , m_rows(rows)
  , m_cols(cols) {}

Matrix::Matrix(DataArray&& data, size_t cols, size_t rows)
  : m_storage(std::move(data))
  , m_data(m_storage.data())
  , m_rows(rows)
  , m_cols(cols) {}

Matrix::Matrix(const Matrix& cpy) {
  *this = cpy;
}

Matrix::Matrix(Matrix&& mv) {
  *this = std::move(mv);
}

Matrix& Matrix::operator=(const Matrix& rhs) {
  m_cols = rhs.m_cols;
  m_rows = rhs.m_rows;

  if (rhs.isShallow()) {
    m_storage = DataArray();
    m_data = rhs.m_data;
  }
  else {
    m_storage = DataArray(size());
    m_data = m_storage.data();
    memcpy(m_data, rhs.m_data, size() * sizeof(double));
  }

  return *this;
}

Matrix& Matrix::operator=(Matrix&& rhs) {
  m_cols = rhs.m_cols;
  m_rows = rhs.m_rows;

  if (rhs.isShallow()) {
    m_storage = DataArray();
    m_data = rhs.m_data;
  }
  else {
    m_storage = std::move(rhs.m_storage);
    m_data = m_storage.data();
  }

  rhs.m_data = nullptr;
  rhs.m_cols = 0;
  rhs.m_rows = 0;

  return *this;
}

Vector Matrix::operator*(const Vector& rhs) const {
  ASSERT(rhs.size() == m_cols);

  Vector v(m_rows);
  for (size_t r = 0; r < m_rows; ++r) {
    double sum = 0.0;
    for (size_t c = 0; c < m_cols; ++c) {
      sum += at(c, r) * rhs[c];
    }
    v[r] = sum;
  }
  return v;
}

Matrix Matrix::operator+(const Matrix& rhs) const {
  ASSERT(rhs.m_cols == m_cols);
  ASSERT(rhs.m_rows == m_rows);

  Matrix m(m_cols, m_rows);

  for (size_t i = 0; i < size(); ++i) {
    m.m_data[i] = m_data[i] + rhs.m_data[i];
  }

  return m;
}

Matrix Matrix::operator-(const Matrix& rhs) const {
  ASSERT(rhs.m_cols == m_cols);
  ASSERT(rhs.m_rows == m_rows);

  Matrix m(m_cols, m_rows);

  for (size_t i = 0; i < size(); ++i) {
    m.m_data[i] = m_data[i] - rhs.m_data[i];
  }

  return m;
}

void Matrix::operator+=(double x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] += x;
  }
}

Vector Matrix::transposeMultiply(const Vector& rhs) const {
  ASSERT(rhs.size() == m_rows);

  Vector v(m_cols);
  for (size_t c = 0; c < m_cols; ++c) {
    double sum = 0.0;
    for (size_t r = 0; r < m_rows; ++r) {
      sum += at(c, r) * rhs[r];
    }
    v[c] = sum;
  }
  return v;
}

void Matrix::zero() {
  memset(m_data, 0, size() * sizeof(double));
}

void Matrix::fill(double x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] = x;
  }
}

void Matrix::randomize(double standardDeviation) {
  std::default_random_engine gen;
  std::normal_distribution<double> dist(0.0, standardDeviation);

  for (size_t i = 0; i < size(); ++i) {
    m_data[i] = dist(gen);
  }
}

double Matrix::sum() const {
  double s = 0.0;

  for (size_t i = 0; i < size(); ++i) {
    s += m_data[i];
  }

  return s;
}

Matrix Matrix::transpose() const {
  Matrix m(m_rows, m_cols);
  for (size_t c = 0; c < m_cols; ++c) {
    for (size_t r = 0; r < m_rows; ++r) {
      m.set(r, c, at(c, r));
    }
  }
  return m;
}

bool Matrix::operator==(const Matrix& rhs) const {
  if (!(m_cols == rhs.m_cols && m_rows == rhs.m_rows)) {
    return false;
  }

  return arraysEqual(m_data, rhs.m_data, size());
}

MatrixPtr Matrix::createShallow(DataArray& data, size_t cols, size_t rows) {
  TRUE_OR_THROW(data.size() == cols * rows, "cols * rows != data length");
  return MatrixPtr(new Matrix(data.data(), cols, rows));
}

ConstMatrixPtr Matrix::createShallow(const DataArray& data, size_t cols, size_t rows) {
  TRUE_OR_THROW(data.size() == cols * rows, "cols * rows != data length");
  return ConstMatrixPtr(new Matrix(const_cast<double*>(data.data()), cols, rows));
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
  os << "[ ";
  for (size_t j = 0; j < m.rows(); ++j) {
    if (j > 0) {
      os << "  ";
    }
    for (size_t i = 0; i < m.cols(); ++i) {
      os << m.at(i, j) << " ";
    }
    if (j + 1 == m.rows()) {
      os << "]";
    }
    os << std::endl;
  }

  return os;
}

Kernel::Kernel(std::initializer_list<std::initializer_list<std::initializer_list<double>>> data)
  : m_storage(count(data))
  , m_data(m_storage.data())
  , m_D(data.size())
  , m_H(data.begin()->size())
  , m_W(data.begin()->begin()->size()) {

  size_t z = 0;
  for (auto plane : data) {
    size_t y = 0;
    for (auto row : plane) {
      size_t x = 0;
      for (double value : row) {
        set(x, y, z, value);
        ++x;
      }
      ++y;
    }
    ++z;
  }
}

Kernel::Kernel(size_t W, size_t H, size_t D)
  : m_storage(W * H * D)
  , m_data(m_storage.data())
  , m_D(D)
  , m_H(H)
  , m_W(W) {}

Kernel::Kernel(const DataArray& data, size_t W, size_t H, size_t D)
  : m_storage(data)
  , m_data(m_storage.data())
  , m_D(D)
  , m_H(H)
  , m_W(W) {}

Kernel::Kernel(DataArray&& data, size_t W, size_t H, size_t D)
  : m_storage(std::move(data))
  , m_data(m_storage.data())
  , m_D(D)
  , m_H(H)
  , m_W(W) {}

Kernel::Kernel(double* data, size_t W, size_t H, size_t D)
  : m_data(data)
  , m_D(D)
  , m_H(H)
  , m_W(W) {}

Kernel::Kernel(const Kernel& cpy) {
  *this = cpy;
}

Kernel::Kernel(Kernel&& mv) {
  *this = std::move(mv);
}

void Kernel::convolve(const Array3& image, Array2& featureMap) const {
  ASSERT(image.W() >= m_W);
  ASSERT(image.H() >= m_H);
  ASSERT(image.D() == m_D);

  size_t fmW = image.W() - m_W + 1;
  size_t fmH = image.H() - m_H + 1;

  ASSERT(featureMap.W() == fmW);
  ASSERT(featureMap.H() == fmH);

  for (size_t fmY = 0; fmY < fmH; ++fmY) {
    for (size_t fmX = 0; fmX < fmW; ++fmX) {
      double sum = 0.0;
      for (size_t k = 0; k < m_D; ++k) {
        for (size_t j = 0; j < m_H; ++j) {
          for (size_t i = 0; i < m_W; ++i) {
            sum += image.at(fmX + i, fmY + j, k) * at(i, j, k);
          }
        }
      }
      featureMap.set(fmX, fmY, sum);
    }
  }
}

bool Kernel::operator==(const Kernel& rhs) const {
  if (!(m_W == rhs.m_W && m_H == rhs.m_H && m_D == rhs.m_D)) {
    return false;
  }

  return arraysEqual(m_data, rhs.m_data, size());
}

Kernel& Kernel::operator=(const Kernel& rhs) {
  m_W = rhs.m_W;
  m_H = rhs.m_H;
  m_D = rhs.m_D;

  if (rhs.isShallow()) {
    m_storage = DataArray();
    m_data = rhs.m_data;
  }
  else {
    m_storage = DataArray(size());
    m_data = m_storage.data();
    memcpy(m_data, rhs.m_data, size() * sizeof(double));
  }

  return *this;
}

Kernel& Kernel::operator=(Kernel&& rhs) {
  m_W = rhs.m_W;
  m_H = rhs.m_H;
  m_D = rhs.m_D;

  if (rhs.isShallow()) {
    m_storage = DataArray();
    m_data = rhs.m_data;
  }
  else {
    m_storage = std::move(rhs.m_storage);
    m_data = m_storage.data();
  }

  rhs.m_data = nullptr;
  rhs.m_W = 0;
  rhs.m_H = 0;
  rhs.m_D = 0;

  return *this;
}

void Kernel::zero() {
  memset(m_data, 0, size() * sizeof(double));
}

void Kernel::fill(double x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] = x;
  }
}

void Kernel::randomize(double standardDeviation) {
  std::default_random_engine gen;
  std::normal_distribution<double> dist(0.0, standardDeviation);

  for (size_t i = 0; i < size(); ++i) {
    m_data[i] = dist(gen);
  }
}

Kernel Kernel::operator+(const Kernel& rhs) const {
  Kernel K(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    K.m_data[i] = m_data[i] + rhs.m_data[i];
  }
  return K;
}

Kernel Kernel::operator-(const Kernel& rhs) const {
  Kernel K(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    K.m_data[i] = m_data[i] - rhs.m_data[i];
  }
  return K;
}

Kernel Kernel::operator+(double x) const {
  Kernel K(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    K.m_data[i] = m_data[i] + x;
  }
  return K;
}

Kernel Kernel::operator-(double x) const {
  Kernel K(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    K.m_data[i] = m_data[i] - x;
  }
  return K;
}

Kernel Kernel::operator*(double x) const {
  Kernel K(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    K.m_data[i] = m_data[i] * x;
  }
  return K;
}

Kernel Kernel::operator/(double x) const {
  Kernel K(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    K.m_data[i] = m_data[i] / x;
  }
  return K;
}

void Kernel::operator+=(double x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] += x;
  }
}

void Kernel::operator-=(double x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] -= x;
  }
}

void Kernel::operator*=(double x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] *= x;
  }
}

void Kernel::operator/=(double x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] /= x;
  }
}

Kernel Kernel::computeTransform(const std::function<double(double)>& f) const {
  Kernel K(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    K.m_data[i] = f(m_data[i]);
  }
  return K;
}

void Kernel::transformInPlace(const std::function<double(double)>& f) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] = f(m_data[i]);
  }
}

KernelPtr Kernel::createShallow(DataArray& data, size_t W, size_t H, size_t D) {
  TRUE_OR_THROW(data.size() == W * H * D, "W * H * D != data length");
  return std::unique_ptr<Kernel>(new Kernel(data.data(), W, H, D));
}

ConstKernelPtr Kernel::createShallow(const DataArray& data, size_t W, size_t H, size_t D) {
  TRUE_OR_THROW(data.size() == W * H * D, "W * H * D != data length");
  return std::unique_ptr<const Kernel>(new Kernel(const_cast<double*>(data.data()), W, H, D));
}

std::ostream& operator<<(std::ostream& os, const Kernel& k) {
  os << "[" << std::endl;

  for (size_t z = 0; z < k.D(); ++z) {
    os << "[ ";
    for (size_t y = 0; y < k.H(); ++y) {
      for (size_t x = 0; x < k.W(); ++x) {
        os << k.at(x, y, z) << " ";
      }
      if (y + 1 == k.H()) {
        os << "]";
      }
      os << std::endl;
    }
    if (z + 1 == k.D()) {
      os << "]";
    }
    os << std::endl;
  }

  return os;
}

