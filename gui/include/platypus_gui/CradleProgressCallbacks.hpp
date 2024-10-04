// CradleProgressCallbacks.h
#pragma once

#include <platypus/CradleFunctions.h>
#include <platypus_gui/ImageWidget.hpp>

class CradleProgressCallbacks : public CradleFunctions::Callbacks
{
public:
    explicit CradleProgressCallbacks(ImageWidget* imageWidget);

    virtual bool progress(int value, int total) const override;

private:
    ImageWidget* m_imageWidget;
};
