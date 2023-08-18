#pragma once

#include <memory>
#include <ostream>

class Vector {
  public:
    Vector(size_t size)
      : m_size(size)
      , m_data(std::make_unique<double[]>(size)) {}

    Vector(Vector&& mv)
      : m_size(mv.m_size)
      , m_data(std::move(mv.m_data)) {}

    Vector(std::initializer_list<double>);

    double dot(const Vector& rhs) const;
    inline size_t size() const;
    inline double& operator[](size_t i) const;

    friend std::ostream& operator<<(std::ostream& os, const Vector& v);

  private:
    size_t m_size;
    std::unique_ptr<double[]> m_data;
};

inline size_t Vector::size() const {
  return m_size;
}

inline double& Vector::operator[](size_t i) const {
  return m_data[i];
}
