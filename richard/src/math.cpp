#include <cstring>
#include <random>
#include <iostream> // TODO
#include "math.hpp"
#include "util.hpp"

Vector::Vector(std::initializer_list<double> elements)
  : m_size(elements.size())
  , m_data(new double[m_size]) {

  size_t i = 0;
  for (double x : elements) {
    m_data[i++] = x;
  }
}

Vector::Vector(Vector&& mv)
  : m_size(mv.m_size)
  , m_data(std::move(mv.m_data)) {

  mv.m_size = 0;
}

Vector::Vector(size_t size)
  : m_size(size)
  , m_data(new double[size]) {}

Vector::Vector(const Vector& cpy)
  : m_size(cpy.m_size)
  , m_data(new double[cpy.m_size]) {

  memcpy(m_data.get(), cpy.m_data.get(), m_size * sizeof(double));
  /*
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] = cpy.m_data[i];
  }*/
}

Vector& Vector::operator=(const Vector& rhs) {
  m_size = rhs.m_size;
  m_data.reset(new double[m_size]);

  memcpy(m_data.get(), rhs.m_data.get(), m_size * sizeof(double));
  /*
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] = rhs.m_data[i];
  }
*/
  return *this;
}

Vector& Vector::operator=(Vector&& rhs) {
  m_size = rhs.m_size;
  m_data = std::move(rhs.m_data);

  rhs.m_size = 0;

  return *this;
}

double Vector::magnitude() const {
  double sqSum = 0.0;
  for (size_t i = 0; i < m_size; ++i) {
    double x = m_data[i];
    sqSum += x * x;
  }
  return sqrt(sqSum);
}

void Vector::zero() {
  memset(m_data.get(), 0, m_size * sizeof(double));
}

void Vector::fill(double x) {
  for (size_t i = 0; i < m_size; ++i) {
    m_data[i] = x;
  }
}

void Vector::randomize(double maxMagnitude) {
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(-maxMagnitude, maxMagnitude);

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
  ASSERT(rhs.size() == m_size);

  double x = 0.0;
  for (size_t i = 0; i < m_size; ++i) {
    x += m_data[i] * rhs[i];
  }
  return x;
}

Vector Vector::hadamard(const Vector& rhs) const {
  ASSERT(rhs.size() == m_size);

  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] * rhs[i];
  }
  return v;
}

std::ostream& operator<<(std::ostream& os, const Vector& v) {
  os << "[ ";
  for (size_t i = 0; i < v.size(); ++i) {
    os << v[i] << " ";
  }
  os << "]\n";

  return os;
}

Vector Vector::operator+(const Vector& rhs) const {
  ASSERT(rhs.size() == m_size);

  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = m_data[i] + rhs[i];
  }
  return v;
}

Vector Vector::operator-(const Vector& rhs) const {
  ASSERT(rhs.size() == m_size);

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

bool Vector::operator==(const Vector& rhs) const {
  if (m_size != rhs.m_size) {
    return false;
  }

  for (size_t i = 0; i < m_size; ++i) {
    if (m_data[i] != rhs.m_data[i]) {
      return false;
    }
  }

  return true;
}

Vector Vector::transform(const std::function<double(double)>& f) const {
  Vector v(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    v[i] = f(m_data[i]);
  }
  return v;
}

Matrix::Matrix(size_t cols, size_t rows)
  : m_cols(cols)
  , m_rows(rows) {

  m_data.reset(new double[cols * rows]);
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

Matrix::Matrix(const Matrix& cpy)
  : m_cols(cpy.m_cols)
  , m_rows(cpy.m_rows) {

  m_data.reset(new double[m_rows * m_cols]);
  memcpy(m_data.get(), cpy.m_data.get(), m_rows * m_cols * sizeof(double));
}

Matrix::Matrix(Matrix&& mv)
  : m_cols(mv.m_cols)
  , m_rows(mv.m_rows)
  , m_data(std::move(mv.m_data)) {

  mv.m_rows = 0;
  mv.m_cols = 0;
}

Matrix& Matrix::operator=(const Matrix& rhs) {
  m_cols = rhs.m_cols;
  m_rows = rhs.m_rows;
  m_data.reset(new double[m_rows * m_cols]);
  memcpy(m_data.get(), rhs.m_data.get(), m_rows * m_cols * sizeof(double));

  return *this;
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
  memset(m_data.get(), 0, m_rows * m_cols * sizeof(double));
}
/*
void Matrix::zeroCol(size_t c) {
  for (size_t r = 0; r < m_rows; ++r) {
    set(c, r, 0);
  }
}

void Matrix::zeroRow(size_t r) {
  memset(m_data.get() + r * m_cols * sizeof(double), 0, m_cols * sizeof(double));
}*/

void Matrix::randomize(double maxMagnitude) {
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(-maxMagnitude, maxMagnitude);

  for (size_t i = 0; i < m_rows * m_cols; ++i) {
    m_data[i] = dist(gen);
  }
}
