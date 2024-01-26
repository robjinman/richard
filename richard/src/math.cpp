#include "math.hpp"
#include "exception.hpp"
#include "utils.hpp"
#include <ostream>
#include <iomanip>
#include <cstring>
#include <random>
#include <algorithm>

namespace richard {
namespace {

const size_t FLOAT_PRECISION = 10;

bool arraysEqual(const netfloat_t* A, const netfloat_t* B, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (A[i] != B[i]) {
      return false;
    }
  }
  
  return true;
}

size_t count(std::initializer_list<std::initializer_list<netfloat_t>> X) {
  [[maybe_unused]] size_t H = X.size();
  DBG_ASSERT(H > 0);
  [[maybe_unused]] size_t W = X.begin()->size();
  DBG_ASSERT(W > 0);

  size_t numElements = 0;

  for ([[maybe_unused]] auto row : X) {
    DBG_ASSERT(row.size() == W);
    numElements += W;
  }

  return numElements;
}

size_t count(std::initializer_list<std::initializer_list<std::initializer_list<netfloat_t>>> X) {
  [[maybe_unused]] size_t H = X.size();
  DBG_ASSERT(H > 0);
  [[maybe_unused]] size_t W = X.begin()->size();
  DBG_ASSERT(W > 0);
  size_t D = X.begin()->begin()->size();
  DBG_ASSERT(D > 0);

  size_t numElements = 0;

  for (auto row : X) {
    DBG_ASSERT(row.size() == W);

    for ([[maybe_unused]] auto zLine : row) {
      DBG_ASSERT(zLine.size() == D);
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
  : m_data(new netfloat_t[size])
  , m_size(size) {

  memset(m_data.get(), 0, m_size * sizeof(netfloat_t));
}

DataArray::DataArray(const DataArray& cpy)
  : m_data(new netfloat_t[cpy.m_size])
  , m_size(cpy.m_size) {

  memcpy(m_data.get(), cpy.m_data.get(), m_size * sizeof(netfloat_t));
}

DataArray::DataArray(DataArray&& mv)
  : m_data(std::move(mv.m_data))
  , m_size(mv.m_size) {

  mv.m_size = 0;
}

DataArray& DataArray::operator=(const DataArray& rhs) {
  m_size = rhs.m_size;
  m_data.reset(new netfloat_t[m_size]);
  memcpy(m_data.get(), rhs.m_data.get(), m_size * sizeof(netfloat_t));

  return *this;
}

DataArray& DataArray::operator=(DataArray&& rhs) {
  m_size = rhs.m_size;
  m_data = std::move(rhs.m_data);

  rhs.m_size = 0;

  return *this;
}

DataArray DataArray::concat(const std::vector<std::reference_wrapper<DataArray>>& arrays) {
  size_t totalSize = 0;
  for (auto array : arrays) {
    totalSize += array.get().size();
  }

  DataArray result(totalSize);

  netfloat_t* ptr = result.m_data.get();
  for (auto array : arrays) {
    memcpy(ptr, array.get().m_data.get(), array.get().size() * sizeof(netfloat_t));
    ptr += array.get().size();
  }

  return result;
}

std::ostream& operator<<(std::ostream& os, const DataArray& v) {
  os << "[ ";
  for (size_t i = 0; i < v.size(); ++i) {
    os << std::fixed << std::setprecision(FLOAT_PRECISION) << v[i] << " ";
  }
  os << "]" << std::endl;

  return os;
}

Vector::Vector()
  : m_data(nullptr)
  , m_size(0) {}

Vector::Vector(std::initializer_list<netfloat_t> data)
  : m_storage(data.size())
  , m_data(m_storage.data())
  , m_size(data.size()) {

  size_t i = 0;
  for (netfloat_t x : data) {
    m_data[i++] = x;
  }
}

Vector::Vector(size_t length)
  : m_storage(length)
  , m_data(m_storage.data())
  , m_size(length) {}

Vector::Vector(netfloat_t* data, size_t size)
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

Vector::Vector(const Vector& cpy)
  : m_storage(cpy.m_size)
  , m_data(m_storage.data())
  , m_size(cpy.m_size) {

  memcpy(m_data, cpy.m_data, m_size * sizeof(netfloat_t));
}

Vector::Vector(Vector&& mv) {
  m_size = mv.m_size;

  if (mv.isShallow()) {
    m_storage = DataArray(m_size);
    m_data = m_storage.data();
    memcpy(m_data, mv.m_data, m_size * sizeof(netfloat_t));
  }
  else {
    m_storage = std::move(mv.m_storage);
    m_data = m_storage.data();
    
    mv.m_data = nullptr;
    mv.m_size = 0;
  }
}

Vector& Vector::operator=(const Vector& rhs) {
  if (isShallow()) {
    DBG_ASSERT(rhs.size() == m_size);
  }
  else {
    m_size = rhs.m_size;
    m_storage = DataArray(m_size);
    m_data = m_storage.data();
  }

  memcpy(m_data, rhs.m_data, m_size * sizeof(netfloat_t));

  return *this;
}

Vector& Vector::operator=(Vector&& rhs) {
  if (isShallow() || rhs.isShallow()) {
    return this->operator=(rhs);
  }

  m_size = rhs.m_size;
  m_storage = std::move(rhs.m_storage);
  m_data = m_storage.data();

  rhs.m_data = nullptr;
  rhs.m_size = 0;

  return *this;
}

bool Vector::operator==(const Vector& rhs) const {
  if (m_size != rhs.m_size) {
    return false;
  }

  return arraysEqual(m_data, rhs.m_data, m_size);
}

netfloat_t Vector::magnitude() const {
  return sqrt(squareMagnitude());
}

netfloat_t Vector::squareMagnitude() const {
  netfloat_t sqSum = 0.0;
  for (size_t i = 0; i < m_size; ++i) {
    netfloat_t x = m_data[i];
    sqSum += x * x;
  }
  return sqSum;
}

void Vector::zero() {
  memset(m_data, 0, m_size * sizeof(netfloat_t));
}

void Vector::fill(netfloat_t x) {
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] = x;
  }
}

Vector& Vector::randomize(netfloat_t standardDeviation) {
  static std::mt19937 gen(0);
  std::normal_distribution<netfloat_t> dist(0.0, standardDeviation);

  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] = dist(gen);
  }
  
  return *this;
}

void Vector::normalize() {
  netfloat_t mag = magnitude();
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] = m_data[i] / mag;
  }
}

netfloat_t Vector::dot(const Vector& rhs) const {
  DBG_ASSERT(rhs.m_size == m_size);

  netfloat_t x = 0.0;
  for (size_t i = 0; i < m_size; ++i) {
    x += m_data[i] * rhs[i];
  }
  return x;
}

Vector Vector::hadamard(const Vector& rhs) const {
  DBG_ASSERT(rhs.m_size == m_size);

  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] * rhs[i];
  }
  return v;
}

Vector Vector::operator+(const Vector& rhs) const {
  DBG_ASSERT(rhs.m_size == m_size);

  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] + rhs[i];
  }
  return v;
}

Vector Vector::operator-(const Vector& rhs) const {
  DBG_ASSERT(rhs.m_size == m_size);

  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] - rhs[i];
  }
  return v;
}

Vector Vector::operator*(netfloat_t s) const {
  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] * s;
  }
  return v;
}

Vector Vector::operator+(netfloat_t s) const {
  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] + s;
  }
  return v;
}

Vector Vector::operator-(netfloat_t s) const {
  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] - s;
  }
  return v;
}

Vector& Vector::operator+=(const Vector& rhs) {
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] += rhs.m_data[i];
  }
  return *this;
}

Vector& Vector::operator-=(const Vector& rhs) {
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] -= rhs.m_data[i];
  }
  return *this;
}

Vector& Vector::operator+=(netfloat_t x) {
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] += x;
  }
  return *this;
}

Vector& Vector::operator-=(netfloat_t x) {
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] -= x;
  }
  return *this;
}

Vector& Vector::operator*=(netfloat_t x) {
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] *= x;
  }
  return *this;
}

Vector& Vector::operator/=(netfloat_t x) {
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] /= x;
  }
  return *this;
}

netfloat_t Vector::sum() const {
  netfloat_t s = 0.0;

  for (size_t i = 0; i < m_size; ++i) {
    s += m_data[i];
  }

  return s;
}

Vector Vector::computeTransform(const std::function<netfloat_t(netfloat_t)>& f) const {
  Vector v(m_size);

  for (size_t i = 0; i < m_size; ++i) {
    v.m_data[i] = f(m_data[i]);
  }

  return v;
}

void Vector::transformInPlace(const std::function<netfloat_t(netfloat_t)>& f) {
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] = f(m_data[i]);
  }
}

VectorPtr Vector::createShallow(DataArray& data) {
  return VectorPtr(new Vector(data.data(), data.size()));
}

ConstVectorPtr Vector::createShallow(const DataArray& data) {
  return ConstVectorPtr(new Vector(const_cast<netfloat_t*>(data.data()), data.size()));
}

VectorPtr Vector::createShallow(netfloat_t* data, size_t size) {
  return VectorPtr(new Vector(data, size));
}

ConstVectorPtr Vector::createShallow(const netfloat_t* data, size_t size) {
  return ConstVectorPtr(new Vector(const_cast<netfloat_t*>(data), size));
}

std::ostream& operator<<(std::ostream& os, const Vector& v) {
  os << "[ ";
  for (size_t i = 0; i < v.size(); ++i) {
    os << std::fixed << std::setprecision(FLOAT_PRECISION) << v[i] << " ";
  }
  os << "]" << std::endl;

  return os;
}

Matrix::Matrix()
  : m_data(nullptr)
  , m_rows(0)
  , m_cols(0) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<netfloat_t>> data)
  : m_storage(count(data))
  , m_data(m_storage.data())
  , m_rows(data.size())
  , m_cols(data.begin()->size()) {

  size_t r = 0;
  for (auto row : data) {
    size_t c = 0;
    for (netfloat_t value : row) {
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

Matrix::Matrix(netfloat_t* data, size_t cols, size_t rows)
  : m_data(data)
  , m_rows(rows)
  , m_cols(cols) {}

Matrix::Matrix(const DataArray& data, size_t cols, size_t rows)
  : m_storage(data)
  , m_data(m_storage.data())
  , m_rows(rows)
  , m_cols(cols) {

  DBG_ASSERT(m_storage.size() == size());  
}

Matrix::Matrix(DataArray&& data, size_t cols, size_t rows)
  : m_storage(std::move(data))
  , m_data(m_storage.data())
  , m_rows(rows)
  , m_cols(cols) {

  DBG_ASSERT(m_storage.size() == size());  
}

Matrix::Matrix(const Matrix& cpy)
  : m_storage(cpy.size())
  , m_data(m_storage.data())
  , m_rows(cpy.m_rows)
  , m_cols(cpy.m_cols) {

  memcpy(m_data, cpy.m_data, m_cols * m_rows * sizeof(netfloat_t));
}

Matrix::Matrix(Matrix&& mv) {
  m_cols = mv.m_cols;
  m_rows = mv.m_rows;

  if (mv.isShallow()) {
    m_storage = DataArray(m_cols * m_rows);
    m_data = m_storage.data();
    memcpy(m_data, mv.m_data, m_cols * m_rows * sizeof(netfloat_t));
  }
  else {
    m_storage = std::move(mv.m_storage);
    m_data = m_storage.data();
    
    mv.m_data = nullptr;
    mv.m_cols = 0;
    mv.m_rows = 0;
  }
}

Matrix& Matrix::operator=(const Matrix& rhs) {
  if (isShallow()) {
    DBG_ASSERT(rhs.m_cols == m_cols && rhs.m_rows == m_rows);
  }
  else {
    m_cols = rhs.m_cols;
    m_rows = rhs.m_rows;
    m_storage = DataArray(m_cols * m_rows);
    m_data = m_storage.data();
  }

  memcpy(m_data, rhs.m_data, m_cols * m_rows * sizeof(netfloat_t));

  return *this;
}

Matrix& Matrix::operator=(Matrix&& rhs) {
  if (isShallow() || rhs.isShallow()) {
    return this->operator=(rhs);
  }

  m_cols = rhs.m_cols;
  m_rows = rhs.m_rows;
  m_storage = std::move(rhs.m_storage);
  m_data = m_storage.data();

  rhs.m_data = nullptr;
  rhs.m_cols = 0;
  rhs.m_rows = 0;

  return *this;
}

Vector Matrix::operator*(const Vector& rhs) const {
  DBG_ASSERT(rhs.size() == m_cols);

  Vector v(m_rows);
  for (size_t r = 0; r < m_rows; ++r) {
    netfloat_t sum = 0.0;
    for (size_t c = 0; c < m_cols; ++c) {
      sum += at(c, r) * rhs[c];
    }
    v[r] = sum;
  }
  return v;
}

Matrix Matrix::operator+(const Matrix& rhs) const {
  DBG_ASSERT(rhs.m_cols == m_cols);
  DBG_ASSERT(rhs.m_rows == m_rows);

  Matrix m(m_cols, m_rows);

  for (size_t i = 0; i < size(); ++i) {
    m.m_data[i] = m_data[i] + rhs.m_data[i];
  }

  return m;
}

Matrix Matrix::operator-(const Matrix& rhs) const {
  DBG_ASSERT(rhs.m_cols == m_cols);
  DBG_ASSERT(rhs.m_rows == m_rows);

  Matrix m(m_cols, m_rows);

  for (size_t i = 0; i < size(); ++i) {
    m.m_data[i] = m_data[i] - rhs.m_data[i];
  }

  return m;
}

Matrix Matrix::operator+(netfloat_t x) const {
  Matrix m(m_cols, m_rows);

  for (size_t i = 0; i < size(); ++i) {
    m.m_data[i] = m_data[i] + x;
  }

  return m;
}

Matrix Matrix::operator-(netfloat_t x) const {
  Matrix m(m_cols, m_rows);

  for (size_t i = 0; i < size(); ++i) {
    m.m_data[i] = m_data[i] - x;
  }

  return m;
}

Matrix Matrix::operator*(netfloat_t x) const {
  Matrix m(m_cols, m_rows);

  for (size_t i = 0; i < size(); ++i) {
    m.m_data[i] = m_data[i] * x;
  }

  return m;
}

Matrix Matrix::operator/(netfloat_t x) const {
  Matrix m(m_cols, m_rows);

  for (size_t i = 0; i < size(); ++i) {
    m.m_data[i] = m_data[i] / x;
  }

  return m;
}

Matrix& Matrix::operator+=(netfloat_t x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] += x;
  }
  return *this;
}

Matrix& Matrix::operator-=(netfloat_t x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] -= x;
  }
  return *this;
}

Matrix& Matrix::operator*=(netfloat_t x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] *= x;
  }
  return *this;
}

Matrix& Matrix::operator/=(netfloat_t x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] /= x;
  }
  return *this;
}

Matrix& Matrix::operator+=(const Matrix& rhs) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] += rhs.m_data[i];
  }
  return *this;
}

Matrix& Matrix::operator-=(const Matrix& rhs) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] -= rhs.m_data[i];
  }
  return *this;
}

Vector Matrix::transposeMultiply(const Vector& rhs) const {
  DBG_ASSERT(rhs.size() == m_rows);

  Vector v(m_cols);
  for (size_t c = 0; c < m_cols; ++c) {
    netfloat_t sum = 0.0;
    for (size_t r = 0; r < m_rows; ++r) {
      sum += at(c, r) * rhs[r];
    }
    v[c] = sum;
  }
  return v;
}

Matrix Matrix::hadamard(const Matrix& rhs) const {
  DBG_ASSERT(rhs.m_cols == m_cols);
  DBG_ASSERT(rhs.m_rows == m_rows);

  Matrix m(m_cols, m_rows);
  for (size_t i = 0; i < size(); ++i) {
    m.m_data[i] = m_data[i] * rhs.m_data[i];
  }
  return m;
}

void Matrix::zero() {
  memset(m_data, 0, size() * sizeof(netfloat_t));
}

void Matrix::fill(netfloat_t x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] = x;
  }
}

Matrix& Matrix::randomize(netfloat_t standardDeviation) {
  static std::mt19937 gen(0);
  std::normal_distribution<netfloat_t> dist(0.0, standardDeviation);

  for (size_t i = 0; i < size(); ++i) {
    m_data[i] = dist(gen);
  }
  
  return *this;
}

netfloat_t Matrix::sum() const {
  netfloat_t s = 0.0;

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
  DBG_ASSERT(data.size() == cols * rows);
  return MatrixPtr(new Matrix(data.data(), cols, rows));
}

ConstMatrixPtr Matrix::createShallow(const DataArray& data, size_t cols, size_t rows) {
  DBG_ASSERT(data.size() == cols * rows);
  return ConstMatrixPtr(new Matrix(const_cast<netfloat_t*>(data.data()), cols, rows));
}

MatrixPtr Matrix::createShallow(netfloat_t* data, size_t cols, size_t rows) {
  return MatrixPtr(new Matrix(data, cols, rows));
}

ConstMatrixPtr Matrix::createShallow(const netfloat_t* data, size_t cols, size_t rows) {
  return ConstMatrixPtr(new Matrix(const_cast<netfloat_t*>(data), cols, rows));
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
  os << "[ ";
  for (size_t j = 0; j < m.rows(); ++j) {
    if (j > 0) {
      os << "  ";
    }
    for (size_t i = 0; i < m.cols(); ++i) {
      os << std::fixed << std::setprecision(FLOAT_PRECISION) << m.at(i, j) << " ";
    }
    if (j + 1 == m.rows()) {
      os << "]";
    }
    os << std::endl;
  }

  return os;
}

Kernel::Kernel()
  : m_data(nullptr)
  , m_D(0)
  , m_H(0)
  , m_W(0) {}

Kernel::Kernel(std::initializer_list<std::initializer_list<std::initializer_list<netfloat_t>>> data)
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
      for (netfloat_t value : row) {
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

Kernel::Kernel(const Size3& shape)
  : m_storage(shape[0] * shape[1] * shape[2])
  , m_data(m_storage.data())
  , m_D(shape[0])
  , m_H(shape[1])
  , m_W(shape[2]) {}

Kernel::Kernel(const DataArray& data, size_t W, size_t H, size_t D)
  : m_storage(data)
  , m_data(m_storage.data())
  , m_D(D)
  , m_H(H)
  , m_W(W) {

  DBG_ASSERT(m_storage.size() == size());    
}

Kernel::Kernel(const DataArray& data, const Size3& shape)
  : m_storage(data)
  , m_data(m_storage.data())
  , m_D(shape[0])
  , m_H(shape[1])
  , m_W(shape[2]) {

  DBG_ASSERT(m_storage.size() == size());    
}

Kernel::Kernel(DataArray& data, const Size3& shape)
  : m_storage(data)
  , m_data(m_storage.data())
  , m_D(shape[0])
  , m_H(shape[1])
  , m_W(shape[2]) {

  DBG_ASSERT(m_storage.size() == size());    
}

Kernel::Kernel(DataArray&& data, size_t W, size_t H, size_t D)
  : m_storage(std::move(data))
  , m_data(m_storage.data())
  , m_D(D)
  , m_H(H)
  , m_W(W) {

  DBG_ASSERT(m_storage.size() == size());    
}

Kernel::Kernel(DataArray&& data, const Size3& shape)
  : m_storage(std::move(data))
  , m_data(m_storage.data())
  , m_D(shape[0])
  , m_H(shape[1])
  , m_W(shape[2]) {

  DBG_ASSERT(m_storage.size() == size());    
}

Kernel::Kernel(netfloat_t* data, size_t W, size_t H, size_t D)
  : m_data(data)
  , m_D(D)
  , m_H(H)
  , m_W(W) {}

Kernel::Kernel(const Kernel& cpy)
  : m_storage(cpy.size())
  , m_data(m_storage.data())
  , m_D(cpy.m_D)
  , m_H(cpy.m_H)
  , m_W(cpy.m_W) {

  memcpy(m_data, cpy.m_data, m_W * m_H * m_D * sizeof(netfloat_t));
}

Kernel::Kernel(Kernel&& mv) {
  m_W = mv.m_W;
  m_H = mv.m_H;
  m_D = mv.m_D;

  if (mv.isShallow()) {
    m_storage = DataArray(m_W * m_H * m_D);
    m_data = m_storage.data();
    memcpy(m_data, mv.m_data, m_W * m_H * m_D * sizeof(netfloat_t));
  }
  else {
    m_storage = std::move(mv.m_storage);
    m_data = m_storage.data();
    
    mv.m_data = nullptr;
    mv.m_W = 0;
    mv.m_H = 0;
    mv.m_D = 0;
  }
}

bool Kernel::operator==(const Kernel& rhs) const {
  if (!(m_W == rhs.m_W && m_H == rhs.m_H && m_D == rhs.m_D)) {
    return false;
  }

  return arraysEqual(m_data, rhs.m_data, size());
}

Kernel& Kernel::operator=(const Kernel& rhs) {
  if (isShallow()) {
    DBG_ASSERT(rhs.m_W == m_W && rhs.m_H == m_H && rhs.m_D == m_D);
  }
  else {
    m_W = rhs.m_W;
    m_H = rhs.m_H;
    m_D = rhs.m_D;
    m_storage = DataArray(m_W * m_H * m_D);
    m_data = m_storage.data();
  }

  memcpy(m_data, rhs.m_data, m_W * m_H * m_D * sizeof(netfloat_t));

  return *this;
}

Kernel& Kernel::operator=(Kernel&& rhs) {
  if (isShallow() || rhs.isShallow()) {
    return this->operator=(rhs);
  }

  m_W = rhs.m_W;
  m_H = rhs.m_H;
  m_D = rhs.m_D;
  m_storage = std::move(rhs.m_storage);
  m_data = m_storage.data();

  rhs.m_data = nullptr;
  rhs.m_W = 0;
  rhs.m_H = 0;
  rhs.m_D = 0;

  return *this;
}

void Kernel::zero() {
  memset(m_data, 0, size() * sizeof(netfloat_t));
}

void Kernel::fill(netfloat_t x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] = x;
  }
}

Kernel& Kernel::randomize(netfloat_t standardDeviation) {
  static std::mt19937 gen(0);
  std::normal_distribution<netfloat_t> dist(0.0, standardDeviation);

  for (size_t i = 0; i < size(); ++i) {
    m_data[i] = dist(gen);
  }

  return *this;
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

Kernel Kernel::operator+(netfloat_t x) const {
  Kernel K(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    K.m_data[i] = m_data[i] + x;
  }
  return K;
}

Kernel Kernel::operator-(netfloat_t x) const {
  Kernel K(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    K.m_data[i] = m_data[i] - x;
  }
  return K;
}

Kernel Kernel::operator*(netfloat_t x) const {
  Kernel K(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    K.m_data[i] = m_data[i] * x;
  }
  return K;
}

Kernel Kernel::operator/(netfloat_t x) const {
  Kernel K(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    K.m_data[i] = m_data[i] / x;
  }
  return K;
}

Kernel Kernel::hadamard(const Kernel& rhs) const {
  DBG_ASSERT(rhs.m_W == m_W);
  DBG_ASSERT(rhs.m_H == m_H);
  DBG_ASSERT(rhs.m_D == m_D);

  Kernel k(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    k.m_data[i] = m_data[i] * rhs.m_data[i];
  }
  return k;
}

Kernel& Kernel::operator+=(netfloat_t x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] += x;
  }
  return *this;
}

Kernel& Kernel::operator-=(netfloat_t x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] -= x;
  }
  return *this;
}

Kernel& Kernel::operator*=(netfloat_t x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] *= x;
  }
  return *this;
}

Kernel& Kernel::operator/=(netfloat_t x) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] /= x;
  }
  return *this;
}

Kernel& Kernel::operator+=(const Kernel& rhs) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] += rhs.m_data[i];
  }
  return *this;
}

Kernel& Kernel::operator-=(const Kernel& rhs) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] -= rhs.m_data[i];
  }
  return *this;
}

Kernel Kernel::computeTransform(const std::function<netfloat_t(netfloat_t)>& f) const {
  Kernel K(m_W, m_H, m_D);
  for (size_t i = 0; i < size(); ++i) {
    K.m_data[i] = f(m_data[i]);
  }
  return K;
}

void Kernel::transformInPlace(const std::function<netfloat_t(netfloat_t)>& f) {
  for (size_t i = 0; i < size(); ++i) {
    m_data[i] = f(m_data[i]);
  }
}

KernelPtr Kernel::createShallow(DataArray& data, size_t W, size_t H, size_t D) {
  DBG_ASSERT(data.size() == W * H * D);
  return std::unique_ptr<Kernel>(new Kernel(data.data(), W, H, D));
}

ConstKernelPtr Kernel::createShallow(const DataArray& data, size_t W, size_t H, size_t D) {
  DBG_ASSERT(data.size() == W * H * D);
  return std::unique_ptr<const Kernel>(new Kernel(const_cast<netfloat_t*>(data.data()), W, H, D));
}

KernelPtr Kernel::createShallow(netfloat_t* data, size_t W, size_t H, size_t D) {
  return std::unique_ptr<Kernel>(new Kernel(data, W, H, D));
}

ConstKernelPtr Kernel::createShallow(const netfloat_t* data, size_t W, size_t H, size_t D) {
  return std::unique_ptr<const Kernel>(new Kernel(const_cast<netfloat_t*>(data), W, H, D));
}

void computeCrossCorrelation(const Array3& image, const Kernel& kernel, Array2& result,
  bool flipKernel) {

  const size_t kD = kernel.D();
  const size_t kH = kernel.H();
  const size_t kW = kernel.W();

  DBG_ASSERT(image.W() >= kW);
  DBG_ASSERT(image.H() >= kH);
  DBG_ASSERT(image.D() == kD);

  const size_t fmW = image.W() - kW + 1;
  const size_t fmH = image.H() - kH + 1;

  DBG_ASSERT(result.W() == fmW);
  DBG_ASSERT(result.H() == fmH);

  for (size_t fmY = 0; fmY < fmH; ++fmY) {
    for (size_t fmX = 0; fmX < fmW; ++fmX) {
      netfloat_t sum = 0.0;

      for (size_t k = 0; k < kD; ++k) {
        for (size_t j = 0; j < kH; ++j) {
          for (size_t i = 0; i < kW; ++i) {
            netfloat_t kPx = flipKernel ? kernel.at(kW - i - 1, kH - j - 1, k) : kernel.at(i, j, k);
            sum += image.at(fmX + i, fmY + j, k) * kPx;
          }
        }
      }

      result.set(fmX, fmY, sum);
    }
  }
}

void computeFullCrossCorrelation(const Array3& image, const Kernel& kernel, Array2& result,
  bool flipKernel) {

  const int kD = static_cast<int>(kernel.D());
  const int kH = static_cast<int>(kernel.H());
  const int kW = static_cast<int>(kernel.W());

  DBG_ASSERT(image.D() == kernel.D());

  const int imW = static_cast<int>(image.W());
  const int imH = static_cast<int>(image.H());

  const int fmW = imW + kW - 1;
  const int fmH = imH + kH - 1;

  DBG_ASSERT(result.W() == static_cast<size_t>(fmW));
  DBG_ASSERT(result.H() == static_cast<size_t>(fmH));

  const int xMin = -kW + 1;
  const int yMin = -kH + 1;

  for (int fmY = 0; fmY < fmH; ++fmY) {
    for (int fmX = 0; fmX < fmW; ++fmX) {
      netfloat_t sum = 0.0;

      for (int k = 0; k < kD; ++k) {
        for (int j = std::max(0, kH - fmY - 1); j < std::min(kH, fmH - fmY); ++j) {
          for (int i = std::max(0, kW - fmX - 1); i < std::min(kW, fmW - fmX); ++i) {
            int imX = xMin + fmX + i;
            int imY = yMin + fmY + j;

            netfloat_t kPx = flipKernel ? kernel.at(kW - i - 1, kH - j - 1, k) : kernel.at(i, j, k);
            sum += image.at(imX, imY, k) * kPx;
          }
        }
      }

      result.set(fmX, fmY, sum);
    }
  }
}

Matrix outerProduct(const Vector& A, const Vector& B) {
  Matrix M(B.size(), A.size());

  for (size_t j = 0; j < A.size(); ++j) {
    for (size_t i = 0; i < B.size(); ++i) {
      M.set(i, j, A[j] * B[i]);
    }
  }

  return M;
}

std::ostream& operator<<(std::ostream& os, const Kernel& k) {
  os << "[" << std::endl;

  for (size_t z = 0; z < k.D(); ++z) {
    os << "[ ";
    for (size_t y = 0; y < k.H(); ++y) {
      for (size_t x = 0; x < k.W(); ++x) {
        os << std::fixed << std::setprecision(FLOAT_PRECISION) << k.at(x, y, z) << " ";
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

}
