#include "math.hpp"

Vector::Vector(std::initializer_list<double> elements)
  : m_size(elements.size())
  , m_data(std::make_unique<double[]>(m_size)) {

  size_t i = 0;
  for (double x : elements) {
    m_data[i++] = x;
  }
}

double Vector::dot(const Vector& rhs) const {
  double x = 0.0;
  for (size_t i = 0; i < m_size; ++i) {
    x += m_data[i] * rhs[i];
  }
  return x;
}

Vector Vector::hadamard(const Vector& rhs) const {
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
