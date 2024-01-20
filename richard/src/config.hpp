#pragma once

#include "exception.hpp"
#include <variant>
#include <map>
#include <vector>
#include <istream>
#include <type_traits>
#include <memory>

namespace richard {

template <typename T> struct is_vector : public std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A>> : public std::true_type {};

class Config {
  public:
    bool contains(const std::string& key) const;
  
    void setObjectArray(const std::string& key, const std::vector<Config>& value);
    std::vector<Config> getObjectArray(const std::string& key) const;
  
    void setObject(const std::string& key, const Config& value);
    Config getObject(const std::string& key) const;

    std::string dump(int indent = -1) const;

    template <class T>
    void setValue(const std::string& key, T value) {
      if constexpr (std::is_integral_v<T>) {
        m_entries[key] = static_cast<long>(value);
      }
      else if constexpr (std::is_floating_point_v<T>) {
        m_entries[key] = static_cast<double>(value);
      }
      else {
        m_entries[key] = value;
      }
    }
  
    template <class T>
    T getValue(const std::string& key) const {
      ASSERT_MSG(m_entries.count(key), "No '" << key << "' value found in config");

      if constexpr (std::is_arithmetic_v<T>) {
        return std::visit([](auto value) {
          if constexpr (std::is_arithmetic_v<decltype(value)>) {
            return static_cast<T>(value);
          }
          return T{};
        }, m_entries.at(key));
      }
      else {
        return std::get<T>(m_entries.at(key));
      }
    }

    template <class T>
    void setArray(const std::string& key, const std::vector<T>& value) {
      if constexpr (std::is_integral_v<T>) {
        std::vector<long> vec;
        for (auto x : value) {
          vec.push_back(static_cast<long>(x));
        }
        m_entries[key] = vec;
      }
      else if constexpr (std::is_floating_point_v<T>) {
        std::vector<double> vec;
        for (auto x : value) {
          vec.push_back(static_cast<double>(x));
        }
        m_entries[key] = vec;
      }
      else {
        m_entries[key] = value;
      }
    }
  
    template <class T>
    std::vector<T> getArray(const std::string& key) const {
      ASSERT_MSG(m_entries.count(key), "No '" << key << "' value found in config");

      if constexpr (std::is_arithmetic_v<T>) {
        return std::visit([](auto value) {
          if constexpr (is_vector<decltype(value)>::value) {
            if constexpr (std::is_arithmetic_v<typename decltype(value)::value_type>) {
              std::vector<T> vec;
              for (auto x : value) {
                vec.push_back(static_cast<T>(x));
              }
              return vec;
            }
          }
          return std::vector<T>{};
        }, m_entries.at(key));
      }
      else {
        return std::get<std::vector<T>>(m_entries.at(key));
      }
    }

    template <class T, size_t N>
    std::array<T, N> getArray(const std::string& key) const {
      ASSERT_MSG(m_entries.count(key), "No '" << key << "' value found in config");

      if constexpr (std::is_arithmetic_v<T>) {
        return std::visit([](auto value) {
          if constexpr (is_vector<decltype(value)>::value) {
            if constexpr (std::is_arithmetic_v<typename decltype(value)::value_type>) {
              std::array<T, N> arr{};
              for (size_t i = 0; i < std::min(N, value.size()); ++i) {
                arr[i] = static_cast<T>(value[i]);
              }
              return arr;
            }
          }
          return std::array<T, N>{};
        }, m_entries.at(key));
      }
      else {
        auto value = std::get<std::vector<T>>(m_entries.at(key));
        std::array<T, N> arr{};
        for (size_t i = 0; i < std::min(N, value.size()); ++i) {
          arr[i] = static_cast<T>(value[i]);
        }
        return arr;
      }
    }

    bool operator==(const Config& rhs) const;
    bool operator!=(const Config& rhs) const;

    friend class ConfigMaker;

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

    std::map<std::string, ConfigValue> m_entries;
};

}
