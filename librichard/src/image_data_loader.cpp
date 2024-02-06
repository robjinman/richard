#include "richard/image_data_loader.hpp"
#include "richard/exception.hpp"
#include <cpputils/bitmap.hpp>
#include <filesystem>

using namespace cpputils;

namespace richard {

ImageDataLoader::ImageDataLoader(const std::string& directoryPath,
  const std::vector<std::string>& labels, const NormalizationParams& normalization,
  size_t fetchSize)
  : DataLoader(fetchSize)
  , m_normalization(normalization)
  , m_directoryPath(directoryPath) {

  ASSERT_MSG(std::filesystem::is_directory(m_directoryPath),
    "'" << m_directoryPath << "' is not a directory");

  std::filesystem::path directory{directoryPath};
  for (const std::string& label : labels) {
    ASSERT_MSG(std::filesystem::is_directory(directory/label),
      "'" << directory/label << "' is not a directory");

    m_iterators.emplace_back(label, std::filesystem::directory_iterator{directory/label});
  }
}

void ImageDataLoader::seekToBeginning() {
  for (auto& i : m_iterators) {
    i.i = std::filesystem::directory_iterator{m_directoryPath/i.label};
  }
}

size_t ImageDataLoader::loadSamples(std::vector<Sample>& samples) {
  size_t samplesLoaded = 0;

  while (samplesLoaded < fetchSize()) {
    size_t numIteratorsFinished = 0;
    for (auto& cursor : m_iterators) {
      if (cursor.i == std::filesystem::directory_iterator{}) {
        ++numIteratorsFinished;
        continue;
      }

      const auto& entry = *cursor.i;
      if (std::filesystem::is_regular_file(entry)) {
        Bitmap image = loadBitmap(entry.path().string());
        size_t imgW = image.size()[0];
        size_t imgH = image.size()[1];
        size_t channels = image.size()[2];

        Array3 v(imgW, imgH, channels);

        for (size_t j = 0; j < imgH; ++j) {
          for (size_t i = 0; i < imgW; ++i) {
            for (size_t k = 0; k < channels; ++k) {
              v.set(i, j, k, normalize(m_normalization, image[j][i][k]));
            }
          }
        }

        samples.emplace_back(cursor.label, v);
        ++samplesLoaded;
      }

      ++cursor.i;

      if (samplesLoaded >= fetchSize()) {
        break;
      }
    }

    if (numIteratorsFinished == m_iterators.size()) {
      break;
    }
  }

  return samplesLoaded;
}

}
