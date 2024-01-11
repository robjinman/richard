#pragma once

#include "exception.hpp"
#include "types.hpp"
#include <memory>
#include <initializer_list>
#include <stdexcept>
#include <functional>

namespace richard {

class DataArray {
  public:
    DataArray();
    explicit DataArray(size_t size);

    DataArray(const DataArray& cpy);
    DataArray(DataArray&& mv);
    DataArray& operator=(const DataArray& cpy);
    DataArray& operator=(DataArray&& mv);

    inline netfloat_t* data();
    inline const netfloat_t* data() const;

    inline size_t size() const;

    inline netfloat_t& operator[](size_t i);
    inline const netfloat_t& operator[](size_t i) const;

    static DataArray concat(const std::vector<std::reference_wrapper<DataArray>>& arrays);

    friend std::ostream& operator<<(std::ostream& os, const DataArray& v);

  private:
    std::unique_ptr<netfloat_t[]> m_data;
    size_t m_size;
};

netfloat_t* DataArray::data() {
  return m_data.get();
}

const netfloat_t* DataArray::data() const {
  return m_data.get();
}

size_t DataArray::size() const {
  return m_size;
}

netfloat_t& DataArray::operator[](size_t i) {
  return m_data.get()[i];
}

const netfloat_t& DataArray::operator[](size_t i) const {
  return m_data.get()[i];
}

class Vector;
using VectorPtr = std::unique_ptr<Vector>;
using ConstVectorPtr = std::unique_ptr<const Vector>;
using Array = Vector;
using ArrayPtr = VectorPtr;
using ConstArrayPtr = ConstVectorPtr;

class Vector {
  public:
    explicit Vector();
    explicit Vector(std::initializer_list<netfloat_t> data);
    explicit Vector(size_t length);
    Vector(const DataArray& data);
    Vector(DataArray&& data);
    Vector(const Vector& cpy);
    Vector(Vector&& mv);

    inline bool isShallow() const;
    inline const DataArray& storage() const;
    inline DataArray& storage();

    inline netfloat_t* data();
    inline const netfloat_t* data() const;

    Vector& operator=(const Vector& rhs);
    Vector& operator=(Vector&& rhs);

    inline netfloat_t& operator[](size_t i);
    inline const netfloat_t& operator[](size_t i) const;
    
    inline size_t size() const;
    
    bool operator==(const Vector& rhs) const;
    inline bool operator!=(const Vector& rhs) const;

    void zero();
    void normalize();
    Vector& randomize(netfloat_t standardDeviation);
    void fill(netfloat_t x);

    netfloat_t sum() const;
    netfloat_t magnitude() const;
    netfloat_t squareMagnitude() const;
    netfloat_t dot(const Vector& rhs) const;

    Vector operator+(const Vector& rhs) const;
    Vector operator-(const Vector& rhs) const;
    Vector hadamard(const Vector& rhs) const;
    Vector operator/(const Vector& rhs) const;

    Vector operator+(netfloat_t x) const;
    Vector operator-(netfloat_t x) const;
    Vector operator*(netfloat_t x) const;
    Vector operator/(netfloat_t x) const;

    Vector& operator+=(const Vector& rhs);
    Vector& operator-=(const Vector& rhs);

    Vector& operator+=(netfloat_t x);
    Vector& operator-=(netfloat_t x);
    Vector& operator*=(netfloat_t x);
    Vector& operator/=(netfloat_t x);

    Vector computeTransform(const std::function<netfloat_t(netfloat_t)>& f) const;
    void transformInPlace(const std::function<netfloat_t(netfloat_t)>& f);

    // Returns shallow Vector
    inline VectorPtr subvector(size_t from, size_t size);
    inline ConstVectorPtr subvector(size_t from, size_t size) const;

    static VectorPtr createShallow(DataArray& data);
    static ConstVectorPtr createShallow(const DataArray& data);
    static VectorPtr createShallow(netfloat_t* data, size_t size);
    static ConstVectorPtr createShallow(const netfloat_t* data, size_t size);

    friend std::ostream& operator<<(std::ostream& os, const Vector& v);

  private:
    // Creates a shallow Vector
    Vector(netfloat_t* data, size_t size);

    DataArray m_storage;
    netfloat_t* m_data;
    size_t m_size;
};

bool Vector::isShallow() const {
  return m_storage.size() == 0 && m_data != nullptr;
}

const DataArray& Vector::storage() const {
  ASSERT_MSG(!isShallow(), "Attempt to retrieve storage of shallow object");
  return m_storage;
}

DataArray& Vector::storage() {
  ASSERT_MSG(!isShallow(), "Attempt to retrieve storage of shallow object");
  return m_storage;
}

netfloat_t* Vector::data() {
  return m_data;
}

const netfloat_t* Vector::data() const {
  return m_data;
}

netfloat_t& Vector::operator[](size_t i) {
  return m_data[i];
}

const netfloat_t& Vector::operator[](size_t i) const {
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
  public:
    explicit Matrix();
    explicit Matrix(std::initializer_list<std::initializer_list<netfloat_t>> data);
    explicit Matrix(size_t cols, size_t rows);
    Matrix(const DataArray& data, size_t cols, size_t rows);
    Matrix(DataArray&& data, size_t cols, size_t rows);
    Matrix(const Matrix& cpy);
    Matrix(Matrix&& mv);

    inline bool isShallow() const;
    inline const DataArray& storage() const;
    inline DataArray& storage();

    inline netfloat_t* data();
    inline const netfloat_t* data() const;

    inline size_t size() const;

    inline netfloat_t at(size_t col, size_t row) const;
    inline void set(size_t col, size_t row, netfloat_t value);

    inline size_t cols() const;
    inline size_t rows() const;
    inline size_t W() const;
    inline size_t H() const;

    Matrix& operator=(const Matrix& rhs);
    Matrix& operator=(Matrix&& rhs);

    Vector operator*(const Vector& rhs) const;

    Matrix operator+(const Matrix& rhs) const;
    Matrix operator-(const Matrix& rhs) const;

    Matrix operator+(netfloat_t x) const;
    Matrix operator-(netfloat_t x) const;
    Matrix operator*(netfloat_t x) const;
    Matrix operator/(netfloat_t x) const;

    Matrix& operator+=(netfloat_t x);
    Matrix& operator-=(netfloat_t x);
    Matrix& operator*=(netfloat_t x);
    Matrix& operator/=(netfloat_t x);

    Matrix& operator+=(const Matrix& rhs);
    Matrix& operator-=(const Matrix& rhs);

    Matrix hadamard(const Matrix& rhs) const;

    Vector transposeMultiply(const Vector& rhs) const;

    void zero();
    void fill(netfloat_t x);
    Matrix& randomize(netfloat_t standardDeviation);

    netfloat_t sum() const;
    Matrix transpose() const;

    Matrix computeTransform(const std::function<netfloat_t(netfloat_t)>& f) const;
    void transformInPlace(const std::function<netfloat_t(netfloat_t)>& f);

    // Returns shallow Vector
    inline VectorPtr slice(size_t row);
    inline ConstVectorPtr slice(size_t row) const;

    bool operator==(const Matrix& rhs) const;
    inline bool operator!=(const Matrix& rhs) const;

    static MatrixPtr createShallow(DataArray& data, size_t cols, size_t rows);
    static ConstMatrixPtr createShallow(const DataArray& data, size_t cols, size_t rows);
    static MatrixPtr createShallow(netfloat_t* data, size_t cols, size_t rows);
    static ConstMatrixPtr createShallow(const netfloat_t* data, size_t cols, size_t rows);

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

  private:
    // Creates a shallow Matrix
    Matrix(netfloat_t* data, size_t cols, size_t rows);

    DataArray m_storage;
    netfloat_t* m_data;
    size_t m_rows;
    size_t m_cols;
};

bool Matrix::isShallow() const {
  return m_storage.size() == 0 && m_data != nullptr;
}

const DataArray& Matrix::storage() const {
  ASSERT_MSG(!isShallow(), "Attempt to retrieve storage of shallow object");
  return m_storage;
}

DataArray& Matrix::storage() {
  ASSERT_MSG(!isShallow(), "Attempt to retrieve storage of shallow object");
  return m_storage;
}

netfloat_t* Matrix::data() {
  return m_data;
}

const netfloat_t* Matrix::data() const {
  return m_data;
}

size_t Matrix::size() const {
  return m_cols * m_rows;
}

netfloat_t Matrix::at(size_t col, size_t row) const {
  return m_data[row * m_cols + col];
}

void Matrix::set(size_t col, size_t row, netfloat_t value) {
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
  return Vector::createShallow(m_data + row * m_cols, m_cols);
}

ConstVectorPtr Matrix::slice(size_t row) const {
  return Vector::createShallow(m_data + row * m_cols, m_cols);
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
    explicit Kernel();
    explicit Kernel(
      std::initializer_list<std::initializer_list<std::initializer_list<netfloat_t>>> data);
    explicit Kernel(size_t W, size_t H, size_t D);
    explicit Kernel(const Size3& shape);
    Kernel(const DataArray& data, size_t W, size_t H, size_t D);
    Kernel(const DataArray& data, const Size3& shape);
    Kernel(DataArray& data, size_t W, size_t H, size_t D);
    Kernel(DataArray& data, const Size3& shape);
    Kernel(DataArray&& data, size_t W, size_t H, size_t D);
    Kernel(DataArray&& data, const Size3& shape);
    Kernel(const Kernel& cpy);
    Kernel(Kernel&& mv);

    inline void setData(DataArray&& data);

    inline bool isShallow() const;
    inline const DataArray& storage() const;
    inline DataArray& storage();

    inline netfloat_t* data();
    inline const netfloat_t* data() const;

    inline size_t size() const;
    inline Size3 shape() const;

    inline netfloat_t at(size_t x, size_t y, size_t z) const;
    inline void set(size_t x, size_t y, size_t z, netfloat_t value);

    inline size_t W() const;
    inline size_t H() const;
    inline size_t D() const;

    Kernel& operator=(const Kernel& rhs);
    Kernel& operator=(Kernel&& rhs);

    void zero();
    void fill(netfloat_t x);
    Kernel& randomize(netfloat_t standardDeviation);

    Kernel hadamard(const Kernel& rhs) const;

    Kernel operator+(const Kernel& rhs) const;
    Kernel operator-(const Kernel& rhs) const;

    Kernel operator+(netfloat_t x) const;
    Kernel operator-(netfloat_t x) const;
    Kernel operator*(netfloat_t x) const;
    Kernel operator/(netfloat_t x) const;

    Kernel& operator+=(netfloat_t x);
    Kernel& operator-=(netfloat_t x);
    Kernel& operator*=(netfloat_t x);
    Kernel& operator/=(netfloat_t x);

    Kernel& operator+=(const Kernel& rhs);
    Kernel& operator-=(const Kernel& rhs);

    Kernel computeTransform(const std::function<netfloat_t(netfloat_t)>& f) const;
    void transformInPlace(const std::function<netfloat_t(netfloat_t)>& f);

    inline MatrixPtr slice(size_t z);
    inline ConstMatrixPtr slice(size_t z) const;

    bool operator==(const Kernel& rhs) const;
    inline bool operator!=(const Kernel& rhs) const;

    static KernelPtr createShallow(DataArray& data, size_t W, size_t H, size_t D);
    inline static KernelPtr createShallow(DataArray& data, const Size3& shape);
    static ConstKernelPtr createShallow(const DataArray& data, size_t W, size_t H, size_t D);
    inline static ConstKernelPtr createShallow(const DataArray& data, const Size3& shape);
    static KernelPtr createShallow(netfloat_t* data, size_t W, size_t H, size_t D);
    inline static KernelPtr createShallow(netfloat_t* data, const Size3& shape);
    static ConstKernelPtr createShallow(const netfloat_t* data, size_t W, size_t H, size_t D);
    inline static ConstKernelPtr createShallow(const netfloat_t* data, const Size3& shape);

    friend std::ostream& operator<<(std::ostream& os, const Kernel& k);

  private:
    // Creates a shallow Kernel
    Kernel(netfloat_t* data, size_t W, size_t H, size_t D);

    DataArray m_storage;
    netfloat_t* m_data;
    size_t m_D;
    size_t m_H;
    size_t m_W;
};

KernelPtr Kernel::createShallow(DataArray& data, const Size3& shape) {
  return Kernel::createShallow(data, shape[0], shape[1], shape[2]);
}

ConstKernelPtr Kernel::createShallow(const DataArray& data, const Size3& shape) {
  return Kernel::createShallow(data, shape[0], shape[1], shape[2]);
}

KernelPtr Kernel::createShallow(netfloat_t* data, const Size3& shape) {
  return Kernel::createShallow(data, shape[0], shape[1], shape[2]);
}

ConstKernelPtr Kernel::createShallow(const netfloat_t* data, const Size3& shape) {
  return Kernel::createShallow(data, shape[0], shape[1], shape[2]);
}

inline void Kernel::setData(DataArray&& data) {
  DBG_ASSERT(data.size() == size());

  m_storage = std::move(data);
  m_data = m_storage.data();
}

bool Kernel::isShallow() const {
  return m_storage.size() == 0 && m_data != nullptr;
}

const DataArray& Kernel::storage() const {
  ASSERT_MSG(!isShallow(), "Attempt to retrieve storage of shallow object");
  return m_storage;
}

DataArray& Kernel::storage() {
  ASSERT_MSG(!isShallow(), "Attempt to retrieve storage of shallow object");
  return m_storage;
}

netfloat_t* Kernel::data() {
  return m_data;
}

const netfloat_t* Kernel::data() const {
  return m_data;
}

size_t Kernel::size() const {
  return m_W * m_H * m_D;
}

Size3 Kernel::shape() const {
  return { m_W, m_H, m_D };
}

netfloat_t Kernel::at(size_t x, size_t y, size_t z) const {
  return m_data[z * m_W * m_H + y * m_W + x];
}

void Kernel::set(size_t x, size_t y, size_t z, netfloat_t value) {
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
  return Matrix::createShallow(m_data + z * m_W * m_H, m_W, m_H);
}

ConstMatrixPtr Kernel::slice(size_t z) const {
  return Matrix::createShallow(m_data + z * m_W * m_H, m_W, m_H);
}

bool Kernel::operator!=(const Kernel& rhs) const {
  return !(*this == rhs);
}

void computeCrossCorrelation(const Array3& image, const Kernel& kernel, Array2& result,
  bool flipKernel = false);

void computeFullCrossCorrelation(const Array3& image, const Kernel& kernel, Array2& result,
  bool flipKernel = false);

inline void computeConvolution(const Array3& image, const Kernel& kernel, Array2& result) {
  computeCrossCorrelation(image, kernel, result, true);
}

inline void computeFullConvolution(const Array3& image, const Kernel& kernel, Array2& result) {
  computeFullCrossCorrelation(image, kernel, result, true);
}

inline void computeCrossCorrelation(const Array2& image, const Matrix& kernel, Array2& result,
  bool flipKernel = false) {

  ConstArray3Ptr pImage = Array3::createShallow(image.data(), image.W(), image.H(), 1);
  ConstArray3Ptr pKernel = Array3::createShallow(kernel.data(), kernel.W(), kernel.H(), 1);
  computeCrossCorrelation(*pImage, *pKernel, result, flipKernel);
}

inline void computeFullCrossCorrelation(const Array2& image, const Matrix& kernel, Array2& result,
  bool flipKernel = false) {

  ConstArray3Ptr pImage = Array3::createShallow(image.data(), image.W(), image.H(), 1);
  ConstArray3Ptr pKernel = Array3::createShallow(kernel.data(), kernel.W(), kernel.H(), 1);
  computeFullCrossCorrelation(*pImage, *pKernel, result, flipKernel);
}

inline void computeConvolution(const Array2& image, const Matrix& kernel, Array2& result) {
  computeCrossCorrelation(image, kernel, result, true);
}

inline void computeFullConvolution(const Array2& image, const Matrix& kernel, Array2& result) {
  computeFullCrossCorrelation(image, kernel, result, true);
}

}
