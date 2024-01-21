#pragma once

#include "exception.hpp"
#include <variant>
#include <map>
#include <vector>
#include <istream>
#include <memory>

namespace richard {

class Config {
  public:
    bool contains(const std::string& key) const;
  
    bool getBoolean(const std::string& key) const;
    long getInteger(const std::string& key) const;
    double getFloat(const std::string& key) const;
    const std::string& getString(const std::string& key) const;
    const std::vector<std::string>& getStringArray(const std::string& key) const;
    Config getObject(const std::string& key) const;
    std::vector<Config> getObjectArray(const std::string& key) const;

    void setBoolean(const std::string& key, bool value);
    void setInteger(const std::string& key, long value);
    void setFloat(const std::string& key, double value);
    void setString(const std::string& key, const std::string& value);
    void setStringArray(const std::string& key, const std::vector<std::string>& value);
    void setObject(const std::string& key, const Config& value);
    void setObjectArray(const std::string& key, const std::vector<Config>& value);

    template<class T = long>
    std::vector<T> getIntegerArray(const std::string& key) const {
      return coerceVectorType<long, T>(getValue<std::vector<long>>(key));
    }

    template<class T = double>
    std::vector<T> getFloatArray(const std::string& key) const {
      return coerceVectorType<double, T>(getValue<std::vector<double>>(key));
    }

    template<class T, size_t N>
    std::array<T, N> getIntegerArray(const std::string& key) const {
      return vectorToArray<long, T, N>(getIntegerArray(key));
    }

    template<class T, size_t N>
    std::array<T, N> getFloatArray(const std::string& key) const {
      return vectorToArray<double, T, N>(getFloatArray(key));
    }

    template<size_t N>
    std::array<std::string, N> getStringArray(const std::string& key) const {
      return vectorToArray<std::string, N>(getStringArray(key));
    }

    template<class T>
    void setIntegerArray(const std::string& key, const std::vector<T>& value) {
      m_entries[key] = coerceVectorType<T, long>(value);
    }

    template<class T>
    void setFloatArray(const std::string& key, const std::vector<T>& value) {
      m_entries[key] = coerceVectorType<T, double>(value);
    }

    std::string dump(int indent = -1) const;

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

    template<class T>
    const T& getValue(const std::string& key) const {
      ASSERT_MSG(m_entries.count(key), "No '" << key << "' value found in config");
      return std::get<T>(m_entries.at(key));
    }

    template<class T, class ALT>
    T getValue(const std::string& key) const {
      ASSERT_MSG(m_entries.count(key), "No '" << key << "' value found in config");
      const auto& entry = m_entries.at(key);
      if (std::holds_alternative<ALT>(entry)) {
        return std::get<ALT>(entry);
      }
      return std::get<T>(entry);
    }

    template<class SRC_TYPE, class DEST_TYPE, size_t N>
    std::array<DEST_TYPE, N> vectorToArray(const std::vector<SRC_TYPE>& vec) const {
      std::array<DEST_TYPE, N> arr{};
      for (size_t i = 0; i < std::min(vec.size(), N); ++i) {
        arr[i] = static_cast<DEST_TYPE>(vec[i]);
      }
      return arr;
    }

    template<class SRC_TYPE, class DEST_TYPE>
    std::vector<DEST_TYPE> coerceVectorType(const std::vector<SRC_TYPE>& srcVec) const {
      std::vector<DEST_TYPE> destVec(srcVec.size());
      for (size_t i = 0; i < srcVec.size(); ++i) {
        destVec[i] = static_cast<DEST_TYPE>(srcVec[i]);
      }
      return destVec;
    }

    std::map<std::string, ConfigValue> m_entries;
};

}
