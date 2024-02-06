#pragma once

#include "richard/exception.hpp"
#include <variant>
#include <map>
#include <vector>
#include <istream>
#include <memory>
#include <type_traits>

namespace richard {

template<class... Ts>
struct Overloaded : Ts... { using Ts::operator()...; };

template<class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

class Config {
  public:
    bool contains(const std::string& key) const;
  
    bool getBoolean(const std::string& key) const;
    const std::string& getString(const std::string& key) const;
    const std::vector<std::string>& getStringArray(const std::string& key) const;
    Config getObject(const std::string& key) const;
    std::vector<Config> getObjectArray(const std::string& key) const;

    void setBoolean(const std::string& key, bool value);
    void setString(const std::string& key, const std::string& value);
    void setStringArray(const std::string& key, const std::vector<std::string>& value);
    void setObject(const std::string& key, const Config& value);
    void setObjectArray(const std::string& key, const std::vector<Config>& value);

    template<class T>
    T getNumber(const std::string& key) const {
      static_assert(std::is_arithmetic_v<T>, "Expected numeric type");
      return std::visit(Overloaded{
        [this](long value) { return static_cast<T>(value); },
        [this](double value) { return static_cast<T>(value); },
        [this](const auto&) { return T{}; }
      }, getEntry(key));
    }

    template<class T>
    std::vector<T> getNumberArray(const std::string& key) const {
      static_assert(std::is_arithmetic_v<T>, "Expected numeric type");
      return std::visit(Overloaded{
        [this](const std::vector<long>& value) { return coerceVectorType<T>(value); },
        [this](const std::vector<double>& value) { return coerceVectorType<T>(value); },
        [this](const auto&) { return std::vector<T>{}; }
      }, getEntry(key));
    }

    template<class T, size_t N>
    std::array<T, N> getNumberArray(const std::string& key) const {
      static_assert(std::is_arithmetic_v<T>, "Expected numeric type");
      return std::visit(Overloaded{
        [this](const std::vector<long>& value) { return vectorToArray<T, N>(value); },
        [this](const std::vector<double>& value) { return vectorToArray<T, N>(value); },
        [this](const auto&) { return std::array<T, N>{}; }
      }, getEntry(key));
    }

    template<size_t N>
    std::array<std::string, N> getStringArray(const std::string& key) const {
      return vectorToArray<std::string, N>(getStringArray(key));
    }

    template<class T>
    void setNumber(const std::string& key, T value) {
      static_assert(std::is_arithmetic_v<T>, "Expected numeric type");
      using StoredType = std::conditional_t<std::is_floating_point_v<T>, double, long>;
      m_entries[key] = static_cast<StoredType>(value);
    }

    template<class T>
    void setNumberArray(const std::string& key, const std::vector<T>& value) {
      static_assert(std::is_arithmetic_v<T>, "Expected numeric type");
      using StoredType = std::conditional_t<std::is_floating_point_v<T>, double, long>;
      m_entries[key] = coerceVectorType<StoredType, T>(value);
    }

    std::string dump(int indent = -1) const;

    bool operator==(const Config& rhs) const;
    bool operator!=(const Config& rhs) const;

    friend struct ConfigMaker;

    static Config fromJson(const std::string& json);
    static Config fromJson(std::istream& stream);

  private:
    using ConfigValue = std::variant<
      bool,
      long,
      double,
      std::string,
      std::shared_ptr<Config>, // TODO: Find way to use unique_ptr
      std::vector<long>,
      std::vector<double>,
      std::vector<std::string>,
      std::vector<Config>
    >;

    const ConfigValue& getEntry(const std::string& key) const;

    template<class T>
    const T& getValue(const std::string& key) const {
      return std::get<T>(getEntry(key));
    }

    template<class DestType, size_t N, class SrcType>
    std::array<DestType, N> vectorToArray(const std::vector<SrcType>& vec) const {
      std::array<DestType, N> arr{};
      for (size_t i = 0; i < std::min(vec.size(), N); ++i) {
        arr[i] = static_cast<DestType>(vec[i]);
      }
      return arr;
    }

    template<class DestType, class SrcType>
    std::vector<DestType> coerceVectorType(const std::vector<SrcType>& srcVec) const {
      std::vector<DestType> destVec(srcVec.size());
      for (size_t i = 0; i < srcVec.size(); ++i) {
        destVec[i] = static_cast<DestType>(srcVec[i]);
      }
      return destVec;
    }

    std::map<std::string, ConfigValue> m_entries;
};

}
