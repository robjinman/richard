#pragma once

#include <memory>
#include <initializer_list>
#include <cassert>
#include <stdexcept>
#include <functional>
#include "exception.hpp"

class DataArray {
  public:
    DataArray();
    explicit DataArray(size_t size);

    DataArray(const DataArray& cpy);
    DataArray(DataArray&& mv);
    DataArray& operator=(const DataArray& cpy);
    DataArray& operator=(DataArray&& mv);

    inline double* data();
    inline const double* data() const;
    inline std::unique_ptr<double[]>& uniquePtr();

    inline size_t size() const;

    inline double& operator[](size_t i);
    inline const double& operator[](size_t i) const;

    static DataArray concat(const DataArray& A, const DataArray& B);

    friend std::ostream& operator<<(std::ostream& os, const DataArray& v);

  private:
    std::unique_ptr<double[]> m_data;
    size_t m_size;
};

double* DataArray::data() {
  return m_data.get();
}

const double* DataArray::data() const {
  return m_data.get();
}

size_t DataArray::size() const {
  return m_size;
}

double& DataArray::operator[](size_t i) {
  return m_data.get()[i];
}

const double& DataArray::operator[](size_t i) const {
  return m_data.get()[i];
}

class Vector;
using VectorPtr = std::unique_ptr<Vector>;
using ConstVectorPtr = std::unique_ptr<const Vector>;
using Array = Vector;
using ArrayPtr = VectorPtr;
using ConstArrayPtr = ConstVectorPtr;

class Vector {
  friend class Matrix;

  public:
    explicit Vector(std::initializer_list<double> data);
    explicit Vector(size_t length);
    Vector(const DataArray& data);
    Vector(DataArray&& data);
    Vector(const Vector& cpy);
    Vector(Vector&& mv);

    inline bool isShallow() const;
    inline const DataArray& storage() const;
    inline DataArray& storage();

    inline double* data();
    inline const double* data() const;

    Vector& operator=(const Vector& rhs);
    Vector& operator=(Vector&& rhs);

    inline double& operator[](size_t i);
    inline const double& operator[](size_t i) const;
    
    inline size_t size() const;
    
    bool operator==(const Vector& rhs) const;
    inline bool operator!=(const Vector& rhs) const;

    void zero();
    void normalize();
    void randomize(double standardDeviation);
    void fill(double x);

    double sum() const;
    double magnitude() const;
    double squareMagnitude() const;
    double dot(const Vector& rhs) const;

    Vector operator+(const Vector& rhs) const;
    Vector operator-(const Vector& rhs) const;
    Vector hadamard(const Vector& rhs) const;
    Vector operator/(const Vector& rhs) const;

    Vector operator+(double x) const;
    Vector operator-(double x) const;
    Vector operator*(double x) const;
    Vector operator/(double x) const;

    void operator+=(double x);
    void operator-=(double x);
    void operator*=(double x);
    void operator/=(double x);

    Vector computeTransform(const std::function<double(double)>& f) const;
    void transformInPlace(const std::function<double(double)>& f);

    // Returns shallow Vector
    inline VectorPtr subvector(size_t from, size_t size);
    inline ConstVectorPtr subvector(size_t from, size_t size) const;

    static VectorPtr createShallow(DataArray& data);
    static ConstVectorPtr createShallow(const DataArray& data);

    friend std::ostream& operator<<(std::ostream& os, const Vector& v);

  private:
    Vector(double* data, size_t size);

    DataArray m_storage;
    double* m_data;
    size_t m_size;
};

bool Vector::isShallow() const {
  return m_storage.size() == 0;
}

const DataArray& Vector::storage() const {
  return m_storage;
}

DataArray& Vector::storage() {
  return m_storage;
}

double* Vector::data() {
  return m_data;
}

const double* Vector::data() const {
  return m_data;
}

double& Vector::operator[](size_t i) {
  return m_data[i];
}

const double& Vector::operator[](size_t i) const {
  return m_data[i];
}

size_t Vector::size() const {
  return m_size;
}

bool Vector::operator!=(const Vector& rhs) const {
  return !(*this == rhs);
}

VectorPtr Vector::subvector(size_t from, size_t size) {
  return VectorPtr(new Vector(m_data + from, size));
}

ConstVectorPtr Vector::subvector(size_t from, size_t size) const {
  return ConstVectorPtr(new Vector(m_data + from, size));
}

class Matrix;
using MatrixPtr = std::unique_ptr<Matrix>;
using ConstMatrixPtr = std::unique_ptr<const Matrix>;
using Array2 = Matrix;
using Array2Ptr = MatrixPtr;
using ConstArray2Ptr = ConstMatrixPtr;

class Matrix {
  friend class Kernel;

  public:
    explicit Matrix(std::initializer_list<std::initializer_list<double>> data);
    Matrix(size_t cols, size_t rows);
    Matrix(const DataArray& data, size_t cols, size_t rows);
    Matrix(DataArray&& data, size_t cols, size_t rows);
    Matrix(const Matrix& cpy);
    Matrix(Matrix&& mv);

    inline bool isShallow() const;
    inline const DataArray& storage() const;
    inline DataArray& storage();

    inline double* data();
    inline const double* data() const;

    inline size_t size() const;

    inline double at(size_t col, size_t row) const;
    inline void set(size_t col, size_t row, double value);

    inline size_t cols() const;
    inline size_t rows() const;
    inline size_t W() const;
    inline size_t H() const;

    Matrix& operator=(const Matrix& rhs);
    Matrix& operator=(Matrix&& rhs);

    Vector operator*(const Vector& rhs) const;

    Matrix operator+(const Matrix& rhs) const;
    Matrix operator-(const Matrix& rhs) const;

    Matrix operator+(double x) const;
    Matrix operator-(double x) const;
    Matrix operator*(double x) const;
    Matrix operator/(double x) const;

    void operator+=(double x);
    void operator-=(double x);
    void operator*=(double x);
    void operator/=(double x);

    Vector transposeMultiply(const Vector& rhs) const;

    void zero();
    void fill(double x);
    void randomize(double standardDeviation);

    double sum() const;
    Matrix transpose() const;

    Matrix computeTransform(const std::function<double(double)>& f) const;
    void transformInPlace(const std::function<double(double)>& f);

    // Returns shallow Vector
    inline VectorPtr slice(size_t row);
    inline ConstVectorPtr slice(size_t row) const;

    bool operator==(const Matrix& rhs) const;
    inline bool operator!=(const Matrix& rhs) const;

    static MatrixPtr createShallow(DataArray& data, size_t cols, size_t rows);
    static ConstMatrixPtr createShallow(const DataArray& data, size_t cols, size_t rows);

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

  private:
    Matrix(double* data, size_t cols, size_t rows);
  
    DataArray m_storage;
    double* m_data;
    size_t m_rows;
    size_t m_cols;
};

bool Matrix::isShallow() const {
  return m_storage.size() == 0;
}

const DataArray& Matrix::storage() const {
  return m_storage;
}

DataArray& Matrix::storage() {
  return m_storage;
}

double* Matrix::data() {
  return m_data;
}

const double* Matrix::data() const {
  return m_data;
}

size_t Matrix::size() const {
  return m_cols * m_rows;
}

double Matrix::at(size_t col, size_t row) const {
  return m_data[row * m_cols + col];
}

void Matrix::set(size_t col, size_t row, double value) {
  m_data[row * m_cols + col] = value;
}

size_t Matrix::cols() const {
  return m_cols;
}

size_t Matrix::rows() const {
  return m_rows;
}

size_t Matrix::W() const {
  return m_cols;
}

size_t Matrix::H() const {
  return m_rows;
}

VectorPtr Matrix::slice(size_t row) {
  return VectorPtr(new Vector(m_data + row * m_cols, m_cols));
}

ConstVectorPtr Matrix::slice(size_t row) const {
  return ConstVectorPtr(new Vector(m_data + row * m_cols, m_cols));
}

bool Matrix::operator!=(const Matrix& rhs) const {
  return !(*this == rhs);
}

class Kernel;
using KernelPtr = std::unique_ptr<Kernel>;
using ConstKernelPtr = std::unique_ptr<const Kernel>;
using Array3 = Kernel;
using Array3Ptr = KernelPtr;
using ConstArray3Ptr = ConstKernelPtr;

class Kernel {
  public:
    explicit Kernel(
      std::initializer_list<std::initializer_list<std::initializer_list<double>>> data);
    Kernel(size_t W, size_t H, size_t D);
    Kernel(const DataArray& data, size_t W, size_t H, size_t D);
    Kernel(DataArray&& data, size_t W, size_t H, size_t D);
    Kernel(const Kernel& cpy);
    Kernel(Kernel&& mv);

    inline void setData(DataArray&& data);

    inline bool isShallow() const;
    inline const DataArray& storage() const;
    inline DataArray& storage();

    inline double* data();
    inline const double* data() const;

    inline size_t size() const;

    inline double at(size_t x, size_t y, size_t z) const;
    inline void set(size_t x, size_t y, size_t z, double value);

    inline size_t W() const;
    inline size_t H() const;
    inline size_t D() const;

    Kernel& operator=(const Kernel& rhs);
    Kernel& operator=(Kernel&& rhs);

    void zero();
    void fill(double x);
    void randomize(double standardDeviation);

    Kernel operator+(const Kernel& rhs) const;
    Kernel operator-(const Kernel& rhs) const;

    Kernel operator+(double x) const;
    Kernel operator-(double x) const;
    Kernel operator*(double x) const;
    Kernel operator/(double x) const;

    void operator+=(double x);
    void operator-=(double x);
    void operator*=(double x);
    void operator/=(double x);

    Kernel computeTransform(const std::function<double(double)>& f) const;
    void transformInPlace(const std::function<double(double)>& f);

    inline MatrixPtr slice(size_t z);
    inline ConstMatrixPtr slice(size_t z) const;

    bool operator==(const Kernel& rhs) const;
    inline bool operator!=(const Kernel& rhs) const;

    void convolve(const Array3& image, Array2& featureMap) const;

    static KernelPtr createShallow(DataArray& data, size_t W, size_t H, size_t D);
    static ConstKernelPtr createShallow(const DataArray& data, size_t W, size_t H, size_t D);

    friend std::ostream& operator<<(std::ostream& os, const Kernel& k);

  private:
    Kernel(double* data, size_t W, size_t H, size_t D);

    DataArray m_storage;
    double* m_data;
    size_t m_D;
    size_t m_H;
    size_t m_W;
};

inline void Kernel::setData(DataArray&& data) {
  ASSERT(data.size() == size());

  m_storage = std::move(data);
  m_data = m_storage.data();
}

bool Kernel::isShallow() const {
  return m_storage.size() == 0;
}

const DataArray& Kernel::storage() const {
  return m_storage;
}

DataArray& Kernel::storage() {
  return m_storage;
}

double* Kernel::data() {
  return m_data;
}

const double* Kernel::data() const {
  return m_data;
}

size_t Kernel::size() const {
  return m_W * m_H * m_D;
}

double Kernel::at(size_t x, size_t y, size_t z) const {
  return m_data[z * m_W * m_H + y * m_W + x];
}

void Kernel::set(size_t x, size_t y, size_t z, double value) {
  m_data[z * m_W * m_H + y * m_W + x] = value;
}

size_t Kernel::W() const {
  return m_W;
}

size_t Kernel::H() const {
  return m_H;
}

size_t Kernel::D() const {
  return m_D;
}

MatrixPtr Kernel::slice(size_t z) {
  return MatrixPtr(new Matrix(m_data + z * m_W * m_H, m_W, m_H));
}

ConstMatrixPtr Kernel::slice(size_t z) const {
  return ConstMatrixPtr(new Matrix(m_data + z * m_W * m_H, m_W, m_H));
}

bool Kernel::operator!=(const Kernel& rhs) const {
  return !(*this == rhs);
}

