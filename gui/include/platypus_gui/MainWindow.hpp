// MainWindow.h
#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP
#include <platypus_gui/utils.hpp>
#include "platypus_gui/ImageWidget.hpp"
#include "platypus_gui/Polygon.hpp"
#include <platypus/CradleFunctions.h>
#include <platypus/TextureRemoval.h>
#include <QMainWindow>
#include <QProgressDialog>
#include <QImage>
#include <QLabel>
#include <QSlider>
#include <QActionGroup>
#include <QGraphicsScene>
#include <QGraphicsView>


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onOpen();
    void onClose();
    void onSave();
    void onSaveAs();
    void onOpenImage();
    void onExport();
    void onExit();
    void onImageChanged();
    void updateMenus();
    void updateUndo();
    void updateSliders();
    void onTool();
    void onToggleOverlay();
    void onToggleResult();
    void onUndoRedoChanged(bool);
    void onUndoRedoText(const QString &);

    void onCommand();
    void onDelete();

    void onTab(int);
    void onSelection();
    void onToolChanged();

    void onDetectCradle();
    void onRemoveCradle();
    void onRemoveTexture();

    void beginProgress(const QString &msg);
    void endProgress();

    void onCancelProgress();
    void onReset();

    void onDone();
    void onCancel();

    void onPolygonAdded(Polygon *poly);
    void onPolygonRemoved(Polygon *poly);
    void onPolygonValueChanged(Polygon *poly, const QString &);

    void onSlider(int);
    void onBeginChange();
    void onEndChange();

    void onHelp();
    void onAbout();

    // Slots to handle signals from ImageWidget
    void onProcessingStarted(const QString &message);
    void onProcessingFinished();
    void onProcessingCanceled();
    void updateProgressBar(int value, int total);

private:
    void createActions();
    void createMenus();
    void createToolbar();
    void createStatusBar();
    void loadImage(const QString &fileName);
    void saveImage(const QString &fileName);
    void detectCradles();
    void removeCradles();
    void removeTexture();

    ImageWidget *imageWidget;
    QLabel *statusLabel;
    QString currentFile;
    QProgressDialog *progressDialog;
    QSlider *zoomSlider;

    QAction *openAct;
    QAction *saveAct;
    QAction *saveAsAct;
    QAction *exitAct;
    QAction *detectCradleAct;
    QAction *removeCradleAct;
    QAction *removeTextureAct;
    QAction *helpAct;
    QAction *aboutAct;

    QMenu *fileMenu;
    QMenu *editMenu;
    QMenu *helpMenu;
    QProgressBar* m_progressBar;
};

#endif // MAINWINDOW_H
