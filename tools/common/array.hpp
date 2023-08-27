#pragma once

#include <stdexcept>
#include <cstring>
#include <initializer_list>

namespace cpputils {

template <typename T, size_t D>
class ContigMultiArray {
  friend class ContigMultiArray<T, D + 1>;

  public:
    T* data;
    const size_t dimensionality = D;

    ContigMultiArray(const ContigMultiArray& cpy) = delete;

    ContigMultiArray(ContigMultiArray&& mv) {
      data = mv.data;
      memcpy(m_size, mv.m_size, sizeof(size_t) * D);
      m_numElements = mv.m_numElements;
      m_stride = mv.m_stride;
      m_isTopLevel = mv.m_isTopLevel;

      mv.data = nullptr;
    }

    ContigMultiArray(std::initializer_list<ContigMultiArray<T, D - 1>> arr)
      : m_isTopLevel(true) {

      m_size[0] = arr.size();
      memcpy(&m_size[1], arr.begin()->m_size, (D - 1) * sizeof(size_t));

      m_numElements = 1;
      for (size_t i = 0; i < D; ++i) {
        m_numElements *= m_size[i];
      }

      m_stride = computeStride();
      data = new T[m_numElements];

      size_t slice = 0;
      for (auto& sub : arr) {
        memcpy(data + slice * m_stride, sub.data, sub.numElements() * sizeof(T));
        ++slice;
      }
    }

    ContigMultiArray(const size_t size[D])
      : data(nullptr),
        m_isTopLevel(true) {

      m_numElements = 1;
      for (size_t i = 0; i < D; ++i) {
        m_size[i] = size[i];
        m_numElements *= size[i];
      }

      data = new T[m_numElements];

      m_stride = computeStride();
    }

    ContigMultiArray(T* data, const size_t size[D], bool isTopLevel = true)
      : data(data),
        m_isTopLevel(isTopLevel) {

      memcpy(m_size, size, sizeof(size_t) * D);

      m_numElements = 1;
      for (size_t i = 0; i < D; ++i) {
        m_numElements *= size[i];
      }

      m_stride = computeStride();
    }

    const size_t* size() const {
      return m_size;
    }

    size_t numElements() const {
      return m_numElements;
    }

    ContigMultiArray<T, D - 1> operator[](size_t slice) const {
      if (slice >= m_size[0]) {
        throw std::runtime_error("Subscript out of range");
      }

      size_t offset = slice * m_stride;

      return ContigMultiArray<T, D - 1>(data + offset, &m_size[1], false);
    }

    ~ContigMultiArray() {
      if (m_isTopLevel) {
        delete[] data;
      }
    }

  private:
    size_t m_size[D];
    size_t m_numElements;
    size_t m_stride;
    bool m_isTopLevel;

    size_t computeStride() const {
      size_t stride = 1;
      for (size_t i = 1; i < D; ++i) {
        stride *= m_size[i];
      }
      return stride;
    }
};

template<typename T>
class ContigMultiArray<T, 1> {
  friend class ContigMultiArray<T, 2>;

  public:
    T* data;

    ContigMultiArray(const ContigMultiArray& cpy) = delete;

    ContigMultiArray(ContigMultiArray&& mv) {
      data = mv.data;
      m_size[0] = mv.m_size[0];
      m_isTopLevel = mv.m_isTopLevel;

      mv.data = nullptr;
    }

    ContigMultiArray(std::initializer_list<T> elems)
      : m_isTopLevel(true) {

      data = new T[elems.size()];

      int i = 0;
      for (auto& e : elems) {
        data[i] = e;
        ++i;
      }

      m_size[0] = elems.size();
    }

    ContigMultiArray(size_t size[1])
      : m_isTopLevel(true) {

      m_size[0] = *size;
      data = new T[m_size[0]];
    }

    ContigMultiArray(T* data, const size_t size[1], bool isTopLevel = true)
      : data(data),
        m_isTopLevel(isTopLevel) {

      memcpy(m_size, size, sizeof(size_t));
    }

    const size_t* size() const {
      return m_size;
    }

    size_t numElements() const {
      return m_size[0];
    }

    const T& operator[](size_t i) const {
      if (i >= m_size[0]) {
        throw std::runtime_error("Subscript out of range");
      }

      return data[i];
    }

    T& operator[](size_t i) {
      if (i >= m_size[0]) {
        throw std::runtime_error("Subscript out of range");
      }

      return data[i];
    }

    ~ContigMultiArray() {
      if (m_isTopLevel) {
        delete[] data;
      }
    }

  private:
    size_t m_size[1];
    bool m_isTopLevel;
};

}
