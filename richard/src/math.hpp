#pragma once

#include <memory>
#include <ostream>
#include <functional>

class Vector {
  public:
    Vector(size_t size)
      : m_size(size)
      , m_data(std::make_unique<double[]>(size)) {}

    Vector(const Vector& cpy);

    Vector(Vector&& mv)
      : m_size(mv.m_size)
      , m_data(std::move(mv.m_data)) {}

    Vector(std::initializer_list<double>);

    Vector& operator=(const Vector& rhs);

    double dot(const Vector& rhs) const;
    Vector hadamard(const Vector& rhs) const;
    Vector operator+(const Vector& rhs) const;
    inline size_t size() const;
    inline double& operator[](size_t i) const;
    Vector transform(const std::function<double(double)>& f) const;

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

class Matrix {
  public:
    Matrix(size_t cols, size_t rows);

    Vector operator*(const Vector& rhs) const;

  private:
    size_t m_cols;
    size_t m_rows;
    std::unique_ptr<double[]> m_data;
};
