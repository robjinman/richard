#include <cstring>
#include "math.hpp"
#include "util.hpp"

Vector::Vector(std::initializer_list<double> elements)
  : m_size(elements.size())
  , m_data(std::make_unique<double[]>(m_size)) {

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
  , m_data(std::make_unique<double[]>(size)) {}

Vector::Vector(const Vector& cpy)
  : m_size(cpy.m_size)
  , m_data(std::make_unique<double[]>(cpy.m_size)) {

  memcpy(m_data.get(), cpy.m_data.get(), m_size * sizeof(double));
}

Vector& Vector::operator=(const Vector& rhs) {
  m_size = rhs.m_size;
  m_data = std::make_unique<double[]>(m_size);

  memcpy(m_data.get(), rhs.m_data.get(), m_size * sizeof(double));

  return *this;
}

Vector& Vector::operator=(Vector&& rhs) {
  m_size = rhs.m_size;
  m_data = std::move(rhs.m_data);

  rhs.m_size = 0;

  return *this;
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
    v[i] += m_data[i];
  }
  return v;
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

  m_data = std::make_unique<double[]>(cols * rows);
}

Vector Matrix::operator*(const Vector& rhs) const {
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

void Matrix::zero() {
  memset(m_data.get(), 0, m_rows * m_cols * sizeof(double));
}
