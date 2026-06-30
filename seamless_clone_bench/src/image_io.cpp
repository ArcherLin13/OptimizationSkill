#include "image_io.h"

#include <cstdint>
#include <fstream>
#include <vector>

namespace {

#pragma pack(push, 1)
struct BmpFileHeader {
    uint16_t type = 0x4D42;
    uint32_t size = 0;
    uint16_t reserved1 = 0;
    uint16_t reserved2 = 0;
    uint32_t offset = 54;
};

struct BmpInfoHeader {
    uint32_t size = 40;
    int32_t width = 0;
    int32_t height = 0;
    uint16_t planes = 1;
    uint16_t bitCount = 24;
    uint32_t compression = 0;
    uint32_t imageSize = 0;
    int32_t xPelsPerMeter = 0;
    int32_t yPelsPerMeter = 0;
    uint32_t clrUsed = 0;
    uint32_t clrImportant = 0;
};
#pragma pack(pop)

bool writeBmp(const std::string& path, const cv::Mat& image) {
    if (image.empty()) {
        return false;
    }

    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 3) {
        bgr = image;
    } else {
        return false;
    }

    const int width = bgr.cols;
    const int height = bgr.rows;
    const int rowStride = ((width * 3 + 3) / 4) * 4;
    const uint32_t pixelDataSize = static_cast<uint32_t>(rowStride * height);

    BmpFileHeader fileHeader;
    BmpInfoHeader infoHeader;
    fileHeader.size = 54 + pixelDataSize;
    infoHeader.width = width;
    infoHeader.height = height;
    infoHeader.imageSize = pixelDataSize;

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    out.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    out.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    std::vector<uchar> row(static_cast<size_t>(rowStride), 0);
    for (int y = height - 1; y >= 0; --y) {
        const uchar* src = bgr.ptr<uchar>(y);
        for (int x = 0; x < width; ++x) {
            row[static_cast<size_t>(x) * 3 + 0] = src[x * 3 + 0];
            row[static_cast<size_t>(x) * 3 + 1] = src[x * 3 + 1];
            row[static_cast<size_t>(x) * 3 + 2] = src[x * 3 + 2];
        }
        out.write(reinterpret_cast<const char*>(row.data()), rowStride);
    }
    return static_cast<bool>(out);
}

bool readBmp(const std::string& path, cv::Mat& image, bool grayscale) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    BmpFileHeader fileHeader;
    BmpInfoHeader infoHeader;
    in.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    in.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));
    if (!in || fileHeader.type != 0x4D42 || infoHeader.compression != 0) {
        return false;
    }
    if (infoHeader.bitCount != 24 && infoHeader.bitCount != 8) {
        return false;
    }

    const int width = infoHeader.width;
    const int height = std::abs(infoHeader.height);
    const bool topDown = infoHeader.height < 0;
    const int channels = infoHeader.bitCount == 8 ? 1 : 3;
    const int rowStride = ((width * channels + 3) / 4) * 4;

    cv::Mat loaded(height, width, channels == 1 ? CV_8UC1 : CV_8UC3);
    std::vector<uchar> row(static_cast<size_t>(rowStride));

    for (int i = 0; i < height; ++i) {
        const int y = topDown ? i : (height - 1 - i);
        in.read(reinterpret_cast<char*>(row.data()), rowStride);
        if (!in) {
            return false;
        }
        uchar* dst = loaded.ptr<uchar>(y);
        if (channels == 1) {
            for (int x = 0; x < width; ++x) {
                dst[x] = row[static_cast<size_t>(x)];
            }
        } else {
            for (int x = 0; x < width; ++x) {
                dst[x * 3 + 0] = row[static_cast<size_t>(x) * 3 + 0];
                dst[x * 3 + 1] = row[static_cast<size_t>(x) * 3 + 1];
                dst[x * 3 + 2] = row[static_cast<size_t>(x) * 3 + 2];
            }
        }
    }

    if (grayscale && loaded.channels() == 3) {
        cv::cvtColor(loaded, image, cv::COLOR_BGR2GRAY);
    } else {
        image = loaded;
    }
    return true;
}

}  // namespace

bool saveImageFile(const std::string& path, const cv::Mat& image) {
    return writeBmp(path, image);
}

bool loadImageFile(const std::string& path, cv::Mat& image, bool grayscale) {
    return readBmp(path, image, grayscale);
}
