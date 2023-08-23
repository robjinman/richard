#pragma once

#include <memory>
#include <ostream>
#include <functional>

class Vector {
  public:
    Vector(size_t size);
    Vector(const Vector& cpy);
    Vector(Vector&& mv);
    Vector(std::initializer_list<double>);

    Vector& operator=(const Vector& rhs);
    Vector& operator=(Vector&& rhs);
    bool operator==(const Vector& rhs) const;
    inline bool operator!=(const Vector& rhs) const;
    void zero();
    void randomize();
    double dot(const Vector& rhs) const;
    Vector hadamard(const Vector& rhs) const;
    Vector operator+(const Vector& rhs) const;
    Vector operator-(const Vector& rhs) const;
    Vector operator*(double s) const;
    double sum() const;
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

inline bool Vector::operator!=(const Vector& rhs) const {
  return !(*this == rhs);
}

class Matrix {
  public:
    Matrix(size_t cols, size_t rows);

    Vector operator*(const Vector& rhs) const;
    Vector transposeMultiply(const Vector& rhs) const;
    inline double at(size_t col, size_t row) const;
    inline void set(size_t col, size_t row, double x);
    void zero();
    void randomize();
    inline size_t rows() const;
    inline size_t cols() const;

  private:
    size_t m_cols;
    size_t m_rows;
    std::unique_ptr<double[]> m_data;
};

inline double Matrix::at(size_t col, size_t row) const {
  return m_data[row * m_rows + col];
}

inline void Matrix::set(size_t col, size_t row, double x) {
  m_data[row * m_rows + col] = x;
}

inline size_t Matrix::rows() const {
  return m_rows;
}

inline size_t Matrix::cols() const {
  return m_cols;
}
