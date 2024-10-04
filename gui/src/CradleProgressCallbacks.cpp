// CradleProgressCallbacks.cpp
#include <platypus_gui/CradleProgressCallbacks.hpp>
#include <QMetaObject>

CradleProgressCallbacks::CradleProgressCallbacks(ImageWidget* imageWidget) : m_imageWidget(imageWidget) {}


bool CradleProgressCallbacks::progress(int value, int total) const
{
    // Use QMetaObject::invokeMethod to ensure thread-safe GUI updates
    bool continueProcessing = true;
    QMetaObject::invokeMethod(m_imageWidget, "handleProgress",
                              Qt::QueuedConnection,
                              Q_ARG(int, value),
                              Q_ARG(int, total),
                              Q_RETURN_ARG(bool, continueProcessing));
    return continueProcessing;
}
