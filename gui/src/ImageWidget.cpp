// ImageWidget.cpp
#include <platypus_gui/utils.hpp>
#include <platypus_gui/CradleProgressCallbacks.hpp>
#include <platypus_gui/ImageWidget.hpp>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QMessageBox>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPolygonItem>
#include <QGraphicsItem>
#include <QDebug>

ImageWidget::ImageWidget(QWidget *parent)
    : QGraphicsView(parent),
      scene(new QGraphicsScene(this)),
      pixmapItem(nullptr),
      m_processingCanceled(false),
      currentPolygon(nullptr),
      drawingPolygon(false),
      currentZoom(1.0f)
{
    setScene(scene);
    scene->addItem(pixmapItem);
    setRenderHint(QPainter::Antialiasing);
    setDragMode(QGraphicsView::ScrollHandDrag);
}

bool ImageWidget::loadImage(const QString &fileName)
{
    image = QImage(fileName);
    if (image.isNull())
        return false;

    resetTransform();
    currentZoom = 1.0f;

    // Convert QImage to cv::Mat
    cv::Mat tmp = cv::imread(fileName.toStdString(), cv::IMREAD_GRAYSCALE);
    if (tmp.empty())
        return false;

    tmp.convertTo(cvImage, CV_32F, 1.0 / 255);
    tmp.convertTo(original_image, CV_32F, 1.0 / 255);
    // Clear any existing data
    mask_.release();
    cradleComponent.release();
    ms = CradleFunctions::MarkedSegments();
    polygons.clear();

    // Remove and delete the old pixmapItem
    if (pixmapItem)
    {
        scene->removeItem(pixmapItem);
        delete pixmapItem;
        pixmapItem = nullptr;
    }

    // Clear the scene
    scene->clear();

    // Create a new pixmapItem with the new image
    pixmapItem = new QGraphicsPixmapItem(QPixmap::fromImage(image));
    scene->addItem(pixmapItem);

    emit imageChanged();

    return true;
}


bool ImageWidget::saveImage(const QString &fileName)
{
    if (image.isNull())
        return false;

    return image.save(fileName);
}

bool ImageWidget::hasImage() const
{
    return !image.isNull();
}


void ImageWidget::detectCradles()
{
    if (cvImage.empty())
        return;
    pp("Detecting cradles...", __LINE__);
    m_processingCanceled = false;
    emit processingStarted("Detecting cradles...");
    // Create an instance of the callbacks
    CradleProgressCallbacks callbacks(this);
    // Set the callbacks
    CradleFunctions::setCallbacks(&callbacks);

    // Use CradleFunctions to detect cradles
        // Get the size of the image
    QSize size(cvImage.cols, cvImage.rows); // cvImage is cv::Mat

    // Create a QImage for the mask
    QImage maskImage(size, QImage::Format_Mono);
    maskImage.fill(0);

    // If you need to perform operations on maskImage, do them here
    // For example, the user might draw on the maskImage in the dialog
    // Assuming the dialog provides the maskImage
    // maskImage = dialog.getMaskImage(); // If the dialog provides it

    // Convert maskImage to 8-bit indexed format
    maskImage = maskImage.convertToFormat(QImage::Format_Indexed8);

    // Create an OpenCV Mat for the mask
    cv::Mat cvMask(cvImage.rows, cvImage.cols, CV_8UC1, cv::Scalar(0));

    // Build the mask
    buildMask(cvMask, maskImage, CradleFunctions::DEFECT);
    pp("Detecting cradles...", __LINE__);
    CradleFunctions::cradledetect(cvImage, cvMask, vrange_, hrange_);
    // Clear the callbacks after use
    CradleFunctions::setCallbacks(nullptr);

    pp("Detecting cradles...", __LINE__);
    if (m_processingCanceled)
    {
        emit processingCanceled();
        return;
    }
    // Now, create mask based on vrange and hrange
    mask_ = cv::Mat(cvImage.size(), CV_8UC1, cv::Scalar(0));
    // Set vertical ranges in mask
    for (size_t i = 0; i < vrange_.size(); i += 2)
    {
        int startCol = vrange_[i];
        int endCol = vrange_[i+1];

        startCol = std::max(0, startCol);
        endCol = std::min(mask_.cols - 1, endCol);

        if (startCol <= endCol)
        {
            cv::Rect rect(startCol, 0, endCol - startCol + 1, mask_.rows);
            mask_(rect).setTo(255);
        }
    }

    // Set horizontal ranges in mask
    for (size_t i = 0; i < hrange_.size(); i += 2)
    {
        int startRow = hrange_[i];
        int endRow = hrange_[i+1];

        startRow = std::max(0, startRow);
        endRow = std::min(mask_.rows - 1, endRow);

        if (startRow <= endRow)
        {
            cv::Rect rect(0, startRow, mask_.cols, endRow - startRow + 1);
            mask_(rect).setTo(255);
        }
    }

    // Debug: Check if the mask has non-zero values
    int nonZeroCount = cv::countNonZero(mask_);
    qDebug() << "detectCradles: Non-zero pixels in mask:" << nonZeroCount;

    // Update the display
    updateDisplay();
    pp("Detecting cradles...", __LINE__);

    emit processingFinished();
}


void ImageWidget::updateDisplay()
{
    cv::Mat overlayImage;
    cvImage.convertTo(overlayImage, CV_8U, 255);
    cv::Mat coloredImage;
    cv::cvtColor(overlayImage, coloredImage, cv::COLOR_GRAY2BGR);

    if (!mask_.empty())
    {
        int nonZeroCount = cv::countNonZero(mask_);
        qDebug() << "updateDisplay: Non-zero pixels in mask:" << nonZeroCount;

        if (nonZeroCount > 0)
        {
            // Create an overlay image filled with zeros
            cv::Mat overlay = cv::Mat::zeros(coloredImage.size(), coloredImage.type());

            // Set overlay to red where the mask is non-zero
            overlay.setTo(cv::Scalar(0, 0, 255), mask_);

            // Blend the overlay with the original image
            double alpha = 0.5; // adjust transparency
            cv::addWeighted(overlay, alpha, coloredImage, 1.0 - alpha, 0, coloredImage);
        }
    }

    QImage resultImage(coloredImage.data, coloredImage.cols, coloredImage.rows, static_cast<int>(coloredImage.step), QImage::Format_RGB888);
    image = resultImage.copy();

    pixmapItem->setPixmap(QPixmap::fromImage(image));

    // Ensure the pixmapItem is updated
    pixmapItem->update();
    scene->update();
    this->viewport()->update();
}

void ImageWidget::removeCradles()
{
    if (cvImage.empty())
        return;

    if (mask_.empty())
    {
        QMessageBox::warning(this, "Error", "No cradles detected. Please run cradle detection first.");
        return;
    }
    m_processingCanceled = false;
    emit processingStarted("Removing cradles...");

    cv::Mat resultImage, cradleImage;

    // Remove cradles using the detected mask and ranges
    CradleFunctions::removeCradle(cvImage, resultImage, cradleImage, mask_, vrange_, hrange_, ms);
    cv::imshow("Result Image (Cradles Only)", resultImage);
    cv::imshow("Cradle Image (Image with Cradles Removed)", cradleImage);
    cv::imshow("Cradle Image2 (Image with Cradles Removed)", cradleImage - cvImage);
    cv::waitKey(0);

    if (m_processingCanceled)
    {
        emit processingCanceled();
        return;
    }

    // Update cvImage with cradleImage instead of resultImage
    cvImage = resultImage.clone();

    // Update the QImage
    cv::Mat displayImage;
    cvImage.convertTo(displayImage, CV_8U, 255);

    image = QImage(displayImage.data, displayImage.cols, displayImage.rows, displayImage.step, QImage::Format_Grayscale8).copy();
    pixmapItem->setPixmap(QPixmap::fromImage(image));

    // Clear mask and polygons as they may no longer be valid
    mask_.release();
    polygons.clear();

    emit imageChanged();
    emit processingFinished();
}



void ImageWidget::removeTexture()
{
    if (cvImage.empty())
        return;
    pp("Removing Textures...", __LINE__);

    m_processingCanceled = false;
    emit processingStarted("Removing texture...");
    pp("Removing Textures...", __LINE__);
    // Remove cradles using the detected mask and ranges
    cv::imshow("Initial Image", cvImage);
    cv::waitKey(0);
       // Use CradleFunctions to detect cradles
        // Get the size of the image
    QSize size(cvImage.cols, cvImage.rows); // cvImage is cv::Mat

    // Create a QImage for the mask
    QImage maskImage(size, QImage::Format_Mono);
    maskImage.fill(0);

    // If you need to perform operations on maskImage, do them here
    // For example, the user might draw on the maskImage in the dialog
    // Assuming the dialog provides the maskImage
    // maskImage = dialog.getMaskImage(); // If the dialog provides it

    // Convert maskImage to 8-bit indexed format
    maskImage = maskImage.convertToFormat(QImage::Format_Indexed8);

    // Create an OpenCV Mat for the mask
    cv::Mat cvMask(cvImage.rows, cvImage.cols, CV_8UC1, cv::Scalar(0));

    // Build the mask
    buildMask(cvMask, maskImage, CradleFunctions::DEFECT);
    CradleFunctions::cradledetect(cvImage, cvMask, vrange_, hrange_);
    mask_ = cvMask.clone();
    cv::imshow("Initial Image", cvImage);
    cv::imshow("Mask", mask_);
    cv::waitKey(0);
    cv::Mat removed_cradle_image, cradleImage;
    CradleFunctions::removeCradle(cvImage, removed_cradle_image, cradleImage, mask_, vrange_, hrange_, ms);

    // Create an instance of the callbacks
    CradleProgressCallbacks callbacks(this);
    // Set the callbacks
    CradleFunctions::setCallbacks(&callbacks);
    cv::Mat resultImage;
    // Remove texture using TextureRemoval functions
    pp("Removing Textures...", __LINE__);
    cv::imshow("Initial Image", cvImage);
    cv::imshow("Cradle removed X-ray", removed_cradle_image);
    cv::imshow("Mask", mask_);
    cv::imshow("Cradle component after separation", cradleImage);
    cv::waitKey(0);

    TextureRemoval::textureRemove(removed_cradle_image, mask_, resultImage, ms);
    CradleFunctions::setCallbacks(nullptr);
    pp("Removing Textures...", __LINE__);
    cv::imshow("Input Image", cvImage);
    cv::imshow("Texture Removal", resultImage);
    cv::imshow("Mask:", mask_);
    cv::waitKey(0);


    if (m_processingCanceled)
    {
        emit processingCanceled();
        return;
    }
    pp("Removing Textures...", __LINE__);

    // Update cvImage and display
    cvImage = resultImage.clone();
    pp("Removing Textures...", __LINE__);

    // Update the QImage
    cv::Mat displayImage;
    cvImage.convertTo(displayImage, CV_8U, 255);
    pp("Removing Textures...", __LINE__);

    image = QImage(displayImage.data, displayImage.cols, displayImage.rows, displayImage.step, QImage::Format_Grayscale8).copy();
    pixmapItem->setPixmap(QPixmap::fromImage(image));
    pp("Removing Textures...", __LINE__);

    emit imageChanged();
    emit processingFinished();
}

void ImageWidget::setZoom(float scale)
{
    if (scale <= 0)
        return;

    resetTransform();
    currentZoom = scale;
    scale *= 1.0f; // Adjust if necessary
    QTransform transform;
    transform.scale(scale, scale);
    setTransform(transform);
}

void ImageWidget::cancelProcessing()
{
    m_processingCanceled = true;
}


void ImageWidget::wheelEvent(QWheelEvent *event)
{
    // Zoom in/out with mouse wheel
    if (event->angleDelta().y() > 0)
        setZoom(currentZoom * 1.1f);
    else
        setZoom(currentZoom / 1.1f);
}

void ImageWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        // Start drawing a new polygon
        QPointF scenePos = mapToSceneCustom(event->pos());
        if (!drawingPolygon)
        {
            currentPolygon = new Polygon();
            currentPolygon->setPen(QPen(Qt::yellow, 2));
            currentPolygon->setBrush(QBrush(QColor(255, 255, 0, 100))); // Semi-transparent yellow
            connect(currentPolygon, &Polygon::polygonValueChanged, this, &ImageWidget::onPolygonValueChanged);
            polygons.append(currentPolygon);
            scene->addItem(currentPolygon);
            drawingPolygon = true;
        }

        QPolygonF poly = currentPolygon->polygon();
        poly << scenePos;
        currentPolygon->setPolygon(poly);
    }
    else if (event->button() == Qt::RightButton)
    {
        // Finish drawing the polygon
        if (drawingPolygon)
        {
            drawingPolygon = false;
            emit polygonAdded(currentPolygon);
            currentPolygon = nullptr;
        }
    }
    else
    {
        QGraphicsView::mousePressEvent(event);
    }
}

void ImageWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (drawingPolygon && currentPolygon)
    {
        // Optionally, update the current polygon with a temporary point
        // This can be used to show a preview of the next line segment
    }
    else
    {
        QGraphicsView::mouseMoveEvent(event);
    }
}

void ImageWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (!drawingPolygon)
    {
        QGraphicsView::mouseReleaseEvent(event);
    }
}

QPointF ImageWidget::mapToSceneCustom(const QPoint &point)
{
    return mapToScene(point);
}

void ImageWidget::onPolygonValueChanged(Polygon *poly, const QString &value)
{
    // Handle changes to the polygon's value
    emit polygonValueChanged(poly, value);
}

bool ImageWidget::isProcessingCanceled() const
{
    return m_processingCanceled;
}

bool ImageWidget::handleProgress(int value, int total)
{
    // Emit signal to update progress bar
    emit progressUpdated(value, total);

    // Return true to continue processing, or false to cancel
    return !m_processingCanceled;
}

bool ImageWidget::handleProgress(int value, int total, bool /* idk */)
{
    // Emit signal to update progress bar
    emit progressUpdated(value, total);

    // Return true to continue processing, or false to cancel
    return !m_processingCanceled;
}
