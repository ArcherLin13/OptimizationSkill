#pragma once

#include "neon_planar_to_cv32fc3.h"

#include <opencv2/core.hpp>

#include <stdexcept>

namespace planar {

inline cv::Rect clampRoi(const cv::Rect& roi, int width, int height) {
    return roi & cv::Rect(0, 0, width, height);
}

// Equivalent to:  cv::Mat crop = rgb_mat(roi);
// but rgb is planar float buffers (full image), not an interleaved Mat.
//
// Output: continuous CV_32FC3, OpenCV BGR order [B,G,R], same pixel content as
//         an interleaved full-image Mat cropped by roi.
//
// r/g/b: pointers to full-image planes, each row has src_stride floats (>= width).
inline cv::Mat cropAabbFromPlanarF32(const float* r, const float* g, const float* b, int width,
                                     int height, int src_stride, const cv::Rect& roi_in) {
    if (!r || !g || !b || width <= 0 || height <= 0 || src_stride < width) {
        throw std::invalid_argument("cropAabbFromPlanarF32: bad plane args");
    }
    const cv::Rect roi = clampRoi(roi_in, width, height);
    if (roi.empty()) {
        return cv::Mat();
    }

    cv::Mat crop(roi.height, roi.width, CV_32FC3);
    const float* r0 = r + roi.y * src_stride + roi.x;
    const float* g0 = g + roi.y * src_stride + roi.x;
    const float* b0 = b + roi.y * src_stride + roi.x;
    neon_planar_rgb_f32_to_cv32fc3(r0, g0, b0, crop.ptr<float>(0), roi.width, roi.height,
                                   src_stride, static_cast<int>(crop.step1()));
    return crop;
}

// Packed planar full image: [R plane | G plane | B plane], each height*src_stride.
inline cv::Mat cropAabbFromPlanarPackedF32(const float* planar, int width, int height,
                                           int src_stride, const cv::Rect& roi_in) {
    if (!planar) {
        throw std::invalid_argument("cropAabbFromPlanarPackedF32: null planar");
    }
    const int plane = height * src_stride;
    return cropAabbFromPlanarF32(planar + 0 * plane, planar + 1 * plane, planar + 2 * plane, width,
                                 height, src_stride, roi_in);
}

// Planar uchar -> crop as CV_8UC3 BGR (no /255).
inline cv::Mat cropAabbFromPlanarU8(const unsigned char* r, const unsigned char* g,
                                    const unsigned char* b, int width, int height, int src_stride,
                                    const cv::Rect& roi_in) {
    if (!r || !g || !b || width <= 0 || height <= 0 || src_stride < width) {
        throw std::invalid_argument("cropAabbFromPlanarU8: bad plane args");
    }
    const cv::Rect roi = clampRoi(roi_in, width, height);
    if (roi.empty()) {
        return cv::Mat();
    }

    cv::Mat crop(roi.height, roi.width, CV_8UC3);
    for (int y = 0; y < roi.height; ++y) {
        const unsigned char* rr = r + (roi.y + y) * src_stride + roi.x;
        const unsigned char* gg = g + (roi.y + y) * src_stride + roi.x;
        const unsigned char* bb = b + (roi.y + y) * src_stride + roi.x;
        unsigned char* out = crop.ptr<unsigned char>(y);
        for (int x = 0; x < roi.width; ++x) {
            out[0] = bb[x];
            out[1] = gg[x];
            out[2] = rr[x];
            out += 3;
        }
    }
    return crop;
}

// Drop-in style helper for warp path that used to take full rgb_mat:
//   old:  cv::Mat crop_aabb = rgb_mat(roi);
//   new:  cv::Mat crop_aabb = cropAabbFromBuffers(r, g, b, w, h, stride, roi);
inline cv::Mat cropAabbFromBuffers(const float* r, const float* g, const float* b, int width,
                                   int height, int src_stride, const cv::Rect& roi) {
    return cropAabbFromPlanarF32(r, g, b, width, height, src_stride, roi);
}

}  // namespace planar
