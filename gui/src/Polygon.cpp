// Polygon.cpp
#include <platypus_gui/Polygon.hpp>
#include <QPen>
#include <QBrush>

Polygon::Polygon(QObject *parent)
    : QObject(parent), QGraphicsPolygonItem()
{
    // Constructor implementation
    setPen(QPen(Qt::blue, 2));
    setBrush(QBrush(Qt::transparent));

    setFlag(QGraphicsItem::ItemIsSelectable);
    setFlag(QGraphicsItem::ItemIsMovable);

    // Emit the polygonAdded signal
    emit polygonAdded(this);
}

Polygon::~Polygon()
{
    // Emit the polygonRemoved signal
    emit polygonRemoved(this);
}

void Polygon::onValueChanged(const QString &value)
{
    // Handle the value change
    // For example, change the color
    if (value == "red")
    {
        setPen(QPen(Qt::red, 2));
    }
    else if (value == "green")
    {
        setPen(QPen(Qt::green, 2));
    }
    else
    {
        setPen(QPen(Qt::blue, 2));
    }

    // Emit the polygonValueChanged signal
    emit polygonValueChanged(this, value);
}
