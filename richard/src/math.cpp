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

Vector::Vector(const Vector& cpy)
  : m_size(cpy.m_size)
  , m_data(std::make_unique<double[]>(cpy.m_size)) {

  memcpy(m_data.get(), cpy.m_data.get(), m_size * sizeof(double));
}

Vector& Vector::operator=(const Vector& rhs) {
  m_size = rhs.m_size;
  m_data = std::make_unique<double[]>(m_size);

  memcpy(m_data.get(), rhs.m_data.get(), m_size * sizeof(double));
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

  // TODO
}

Vector Matrix::operator*(const Vector& rhs) const {
  // TODO
}
