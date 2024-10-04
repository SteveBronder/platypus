#pragma once
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <QImage>

template <typename T>
void debug_fun(const T& x, int line_num) {
    qDebug() << "Line: " << line_num << " " << x;
}

template <typename T>
void pp(const T& x, int line_num) {
    qDebug() << "Line: " << line_num << " " << x;
}

inline void buildMask(cv::Mat &result, const QImage &mask, unsigned char bitMask)
{
    for (int y = 0; y < result.rows; y++)
    {
        uchar *drow = result.ptr<uchar>(y);
        const uchar *srow = mask.scanLine(y);
        for (int x = 0; x < result.cols; x++)
        {
            if (srow[x])
                drow[x] |= bitMask;
        }
    }
}
