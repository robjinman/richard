#include "util.hpp"
#include "exception.hpp"

namespace richard {

// TODO: Turn into function template that calls get<T>()
nlohmann::json getOrThrow(const nlohmann::json& obj, const std::string& key){
  if (!obj.contains(key)) {
    EXCEPTION("Expected '" << key << "' key in JSON object");
  }
  return obj[key];
}

}
