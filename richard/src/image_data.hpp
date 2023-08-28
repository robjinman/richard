#pragma once

#include <string>

class Dataset;

void loadImageData(Dataset& data, const std::string& directoryPath, const std::string& label);
