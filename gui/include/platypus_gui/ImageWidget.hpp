// ImageWidget.hpp
#ifndef IMAGEWIDGET_HPP
#define IMAGEWIDGET_HPP

#include "platypus_gui/Polygon.hpp"
#include <platypus/CradleFunctions.h>
#include <platypus/TextureRemoval.h>
#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsPolygonItem>
#include <opencv2/opencv.hpp>

class ImageWidget : public QGraphicsView
{
    Q_OBJECT

public:
    explicit ImageWidget(QWidget *parent = nullptr);
    bool loadImage(const QString &fileName);
    bool saveImage(const QString &fileName);
    bool hasImage() const;
    void detectCradles();
    void removeCradles();
    void removeTexture();
    void setZoom(float scale);
    void cancelProcessing();
    bool isProcessingCanceled() const;
public slots:
    bool handleProgress(int value, int total);
    bool handleProgress(int value, int total, bool /* idk */);
signals:
    void imageChanged();
    void processingStarted(const QString &message);
    void processingFinished();
    void processingCanceled();

    // Signals related to polygons
    void polygonAdded(Polygon *poly);
    void polygonRemoved(Polygon *poly);
    void polygonValueChanged(Polygon *poly, const QString &value);
    void progressUpdated(int value, int total);
protected:
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private slots:
    void onPolygonValueChanged(Polygon *poly, const QString &value);

private:
    void updateDisplay();
    QPointF mapToSceneCustom(const QPoint &point);

    QGraphicsScene *scene;
    QGraphicsPixmapItem *pixmapItem;
    QImage image;
    cv::Mat original_image;
    cv::Mat cvImage;

    // Cradle detection and removal data
    cv::Mat mask_;
    CradleFunctions::MarkedSegments ms;
    cv::Mat cradleComponent;
    std::vector<int> vrange_;
    std::vector<int> hrange_;
    // Polygon management
    QList<Polygon *> polygons;
    Polygon *currentPolygon;
    bool drawingPolygon;
    bool m_processingCanceled;
    float currentZoom;

    // Additional functions and members as needed
};

#endif // IMAGEWIDGET_HPP
