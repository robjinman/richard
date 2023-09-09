#include <filesystem>
#include "image_data_set.hpp"
#include "bitmap.hpp"
#include "exception.hpp"

using namespace cpputils;

ImageDataSet::ImageDataSet(const std::string& directoryPath, const std::vector<std::string>& labels)
  : LabelledDataSet(labels)
  , m_directoryPath(directoryPath)
  , m_stats(nullptr) {

  TRUE_OR_THROW(std::filesystem::is_directory(m_directoryPath),
    "'" << m_directoryPath << "' is not a directory");

  std::filesystem::path directory{directoryPath};
  for (const std::string& label : labels) {
    TRUE_OR_THROW(std::filesystem::is_directory(directory/label),
      "'" << directory/label << "' is not a directory");

    m_iterators.emplace_back(label, std::filesystem::directory_iterator{directory/label});
  }
}

const DataStats& ImageDataSet::stats() const {
  return *m_stats;
}

void ImageDataSet::seekToBeginning() {
  for (auto& i : m_iterators) {
    i.i = std::filesystem::directory_iterator{m_directoryPath/i.label};
  }
}

size_t ImageDataSet::loadSamples(std::vector<Sample>& samples, size_t N) {
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

        if (m_stats == nullptr) {
          m_stats = std::make_unique<DataStats>(Vector(v.size()), Vector(v.size()));
          m_stats->min.fill(std::numeric_limits<double>::max());
          m_stats->max.fill(std::numeric_limits<double>::min());
        }

        for (size_t i = 0; i < image.numElements(); ++i) {
          v[i] = static_cast<double>(image.data[i]);

          if (v[i] < m_stats->min[i]) {
            m_stats->min[i] = v[i];
          }
          if (v[i] > m_stats->max[i]) {
            m_stats->max[i] = v[i];
          }
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
