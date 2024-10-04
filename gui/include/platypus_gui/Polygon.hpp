// Polygon.h
#ifndef POLYGON_H
#define POLYGON_H

#include <QObject>
#include <QGraphicsPolygonItem>

class Polygon : public QObject, public QGraphicsPolygonItem
{
    Q_OBJECT

public:
    Polygon(QObject *parent = nullptr);
    ~Polygon();

signals:
    void polygonAdded(Polygon *poly);
    void polygonRemoved(Polygon *poly);
    void polygonValueChanged(Polygon *poly, const QString &value);

public slots:
    void onValueChanged(const QString &value);
};

#endif // POLYGON_H
