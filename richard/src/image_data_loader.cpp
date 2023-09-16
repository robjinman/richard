#include <filesystem>
#include "image_data_loader.hpp"
#include "bitmap.hpp"
#include "exception.hpp"

using namespace cpputils;

ImageDataLoader::ImageDataLoader(const std::string& directoryPath,
  const std::vector<std::string>& labels)
  : m_directoryPath(directoryPath) {

  TRUE_OR_THROW(std::filesystem::is_directory(m_directoryPath),
    "'" << m_directoryPath << "' is not a directory");

  std::filesystem::path directory{directoryPath};
  for (const std::string& label : labels) {
    TRUE_OR_THROW(std::filesystem::is_directory(directory/label),
      "'" << directory/label << "' is not a directory");

    m_iterators.emplace_back(label, std::filesystem::directory_iterator{directory/label});
  }
}

void ImageDataLoader::seekToBeginning() {
  for (auto& i : m_iterators) {
    i.i = std::filesystem::directory_iterator{m_directoryPath/i.label};
  }
}

size_t ImageDataLoader::loadSamples(std::vector<Sample>& samples, size_t N) {
  size_t samplesLoaded = 0;

  while (samplesLoaded < N) {
    size_t numIteratorsFinished = 0;
    for (auto& cursor : m_iterators) {
      if (cursor.i == std::filesystem::directory_iterator{}) {
        ++numIteratorsFinished;
        continue;
      }

      const auto& entry = *cursor.i;
      if (std::filesystem::is_regular_file(entry)) {
        Bitmap image = loadBitmap(entry.path().string());
        Vector v(image.numElements());

        for (size_t i = 0; i < image.numElements(); ++i) {
          v[i] = static_cast<double>(image.data[i]);
        }

        samples.emplace_back(cursor.label, v);
        ++samplesLoaded;
      }

      if (samplesLoaded >= N) {
        break;
      }

      ++cursor.i;
    }

    if (numIteratorsFinished == m_iterators.size()) {
      break;
    }
  }

  return samplesLoaded;
}
