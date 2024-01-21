#include "config.hpp"
#include <nlohmann/json.hpp>

namespace richard {

namespace {

template<class... Ts>
struct Overloaded : Ts... { using Ts::operator()...; };

template<class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

}

struct ConfigMaker {
  static Config fromJson(const nlohmann::json& json);
  static Config::ConfigValue valueFromJsonArray(const nlohmann::json& obj);
  static nlohmann::json toJsonObj(const Config& config);
  static std::vector<Config> createConfigArray(const nlohmann::json& obj);
  static bool isArrayOfObjects(const nlohmann::json& obj);
};

std::vector<Config> ConfigMaker::createConfigArray(const nlohmann::json& obj) {
  std::vector<Config> configs;
  for (auto i : obj) {
    configs.push_back(fromJson(i));
  }
  return configs;
}

bool ConfigMaker::isArrayOfObjects(const nlohmann::json& obj) {
  return obj.type() == nlohmann::json::value_t::array
      && obj.size() > 0
      && obj[0].type() == nlohmann::json::value_t::object;
}

nlohmann::json ConfigMaker::toJsonObj(const Config& config) {
  nlohmann::json obj;
  for (const auto& entry : config.m_entries) {
    std::visit(Overloaded{
      [&](std::shared_ptr<Config> child) {
        obj[entry.first] = toJsonObj(*child);
      },
      [&](std::vector<Config> children) {
        std::vector<nlohmann::json> objs;
        for (const auto& child : children) {
          objs.push_back(toJsonObj(child));
        }
        obj[entry.first] = objs;
      },
      [&](bool value) { obj[entry.first] = value; },
      [&](long value) { obj[entry.first] = value; },
      [&](double value) { obj[entry.first] = value; },
      [&](std::string value) { obj[entry.first] = value; },
      [&](std::vector<long> value) { obj[entry.first] = value; },
      [&](std::vector<double> value) { obj[entry.first] = value; },
      [&](std::vector<std::string> value) { obj[entry.first] = value; },
    }, entry.second);
  }
  return obj;
}

Config::ConfigValue ConfigMaker::valueFromJsonArray(const nlohmann::json& obj) {
  Config::ConfigValue value;

  ASSERT_MSG(obj.size() > 0, "Array is empty");

  auto elementType = obj[0].type();

  switch (elementType) {
    case nlohmann::json::value_t::number_integer:
    case nlohmann::json::value_t::number_unsigned: {
      value = obj.get<std::vector<long>>();
      break;
    }
    case nlohmann::json::value_t::number_float: {
      value = obj.get<std::vector<double>>();
      break;
    }
    case nlohmann::json::value_t::string: {
      value = obj.get<std::vector<std::string>>();
      break;
    }
    default: {
      EXCEPTION("Unsupported type in JSON array");
    }
  }

  return value;
}

Config ConfigMaker::fromJson(const nlohmann::json& obj) {
  Config config;

  for (auto i = obj.begin(); i != obj.end(); ++i) {
    switch (i.value().type()) {
      case nlohmann::json::value_t::boolean: {
        config.setValue<bool>(i.key(), i.value());
        break;
      }
      case nlohmann::json::value_t::number_integer: {
        config.setValue<int>(i.key(), i.value());
        break;
      }
      case nlohmann::json::value_t::number_unsigned: {
        config.setValue<size_t>(i.key(), i.value());
        break;
      }
      case nlohmann::json::value_t::number_float: {
        config.setValue<double>(i.key(), i.value());
        break;
      }
      case nlohmann::json::value_t::string: {
        config.setValue<std::string>(i.key(), i.value());
        break;
      }
      case nlohmann::json::value_t::object: {
        config.setObject(i.key(), fromJson(i.value()));
        break;
      }
      case nlohmann::json::value_t::array: {
        if (isArrayOfObjects(i.value())) {
          config.setObjectArray(i.key(), createConfigArray(i.value()));
        }
        else {
          config.m_entries[i.key()] = valueFromJsonArray(i.value());
        }
        break;
      }
      default: {
        EXCEPTION("Unsupported JSON type");
      }
    }
  }

  return config;
}

bool Config::contains(const std::string& key) const {
  return m_entries.count(key) != 0;
}

void Config::setObjectArray(const std::string& key, const std::vector<Config>& value) {
  m_entries[key] = value;
}

std::vector<Config> Config::getObjectArray(const std::string& key) const {
  ASSERT_MSG(m_entries.count(key), "No '" << key << "' value found in config");
  return std::get<std::vector<Config>>(m_entries.at(key));
}

void Config::setObject(const std::string& key, const Config& value) {
  m_entries[key] = std::make_shared<Config>(value);
}

Config Config::getObject(const std::string& key) const {
  ASSERT_MSG(m_entries.count(key), "No '" << key << "' value found in config");
  return *std::get<std::shared_ptr<Config>>(m_entries.at(key));
}

std::string Config::dump(int indent) const {
  nlohmann::json obj = ConfigMaker::toJsonObj(*this);
  return obj.dump(indent);
}

Config Config::fromJson(const std::string& json) {
  auto obj = nlohmann::json::parse(json);
  return ConfigMaker::fromJson(obj);
}

Config Config::fromJson(std::istream& stream) {
  auto obj = nlohmann::json::parse(stream);
  return ConfigMaker::fromJson(obj);
}

bool Config::operator==(const Config& rhs) const {
  for (const auto& entry : m_entries) {
    if (rhs.m_entries.count(entry.first) == 0) {
      return false;
    }
    auto rhsEntry = rhs.m_entries.at(entry.first);
    bool match = std::visit(Overloaded{
      [&](std::shared_ptr<Config> child) {
        if (!std::holds_alternative<std::shared_ptr<Config>>(rhsEntry)) {
          return false;
        }
        return (*std::get<std::shared_ptr<Config>>(rhsEntry) == *child);
      },
      [&](std::vector<Config> children) {
        if (!std::holds_alternative<std::vector<Config>>(rhsEntry)) {
          return false;
        }
        auto rhsChildren = std::get<std::vector<Config>>(rhsEntry);
        if (children != rhsChildren) {
          return false;
        }
        return true;
      },
      [&](bool value) {
        return std::holds_alternative<bool>(rhsEntry)
          && std::get<bool>(rhsEntry) == value;
      },
      [&](long value) {
        return std::holds_alternative<long>(rhsEntry)
          && std::get<long>(rhsEntry) == value;
      },
      [&](double value) {
        return std::holds_alternative<double>(rhsEntry)
          && std::get<double>(rhsEntry) == value;
      },
      [&](std::string value) {
        return std::holds_alternative<std::string>(rhsEntry)
          && std::get<std::string>(rhsEntry) == value;
      },
      [&](std::vector<long> value) {
        return std::holds_alternative<std::vector<long>>(rhsEntry)
          && std::get<std::vector<long>>(rhsEntry) == value;
      },
      [&](std::vector<double> value) {
        return std::holds_alternative<std::vector<double>>(rhsEntry)
          && std::get<std::vector<double>>(rhsEntry) == value;
      },
      [&](std::vector<std::string> value) {
        return std::holds_alternative<std::vector<std::string>>(rhsEntry)
          && std::get<std::vector<std::string>>(rhsEntry) == value;
      },
    }, entry.second);
    if (!match) {
      return false;
    }
  }
  return true;
}

bool Config::operator!=(const Config& rhs) const {
  return !(*this == rhs);
}

}

