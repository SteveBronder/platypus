// MainWindow.cpp
#include <platypus_gui/MainWindow.hpp>
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QProgressBar>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QAction>
#include <QTabWidget>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      imageWidget(new ImageWidget(this)),
      progressDialog(nullptr)
{
    setWindowTitle("Art Restoration Tool");
    // Initialize progress bar
    m_progressBar = new QProgressBar(this);
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);

    // Assuming you have a status bar
    statusBar()->addPermanentWidget(m_progressBar);

    createActions();
    createMenus();
    createToolbar();
    createStatusBar();

    setCentralWidget(imageWidget);

    resize(800, 600);

    // Connect signals from ImageWidget
    // In MainWindow.cpp constructor or initialization code
    connect(imageWidget, &ImageWidget::polygonAdded, this, &MainWindow::onPolygonAdded);
    connect(imageWidget, &ImageWidget::polygonRemoved, this, &MainWindow::onPolygonRemoved);
    connect(imageWidget, &ImageWidget::polygonValueChanged, this, &MainWindow::onPolygonValueChanged);
    connect(imageWidget, &ImageWidget::processingStarted, this, &MainWindow::onProcessingStarted);
    connect(imageWidget, &ImageWidget::processingFinished, this, &MainWindow::onProcessingFinished);
    connect(imageWidget, &ImageWidget::progressUpdated, this, &MainWindow::updateProgressBar);
    // Connect signals from Polygon
    // Assuming Polygons are managed within ImageWidget, and signals are forwarded appropriately
    // If you create Polygons in MainWindow, connect their signals here
}

void MainWindow::updateProgressBar(int value, int total)
{
    if (total > 0)
    {
        m_progressBar->setMaximum(total);
        m_progressBar->setValue(value);
    }
    else
    {
        m_progressBar->setMaximum(0); // Indeterminate state
    }
}

MainWindow::~MainWindow()
{
    // Clean up if necessary
}

void MainWindow::createActions()
{
    // Create actions and connect them to slots

    openAct = new QAction("&Open Image...", this);
    connect(openAct, &QAction::triggered, this, &MainWindow::onOpen);

    saveAct = new QAction("&Save", this);
    connect(saveAct, &QAction::triggered, this, &MainWindow::onSave);

    saveAsAct = new QAction("Save &As...", this);
    connect(saveAsAct, &QAction::triggered, this, &MainWindow::onSaveAs);

    exitAct = new QAction("E&xit", this);
    connect(exitAct, &QAction::triggered, this, &MainWindow::onExit);

    detectCradleAct = new QAction("&Detect Cradles", this);
    connect(detectCradleAct, &QAction::triggered, this, &MainWindow::onDetectCradle);

    removeCradleAct = new QAction("&Remove Cradles", this);
    connect(removeCradleAct, &QAction::triggered, this, &MainWindow::onRemoveCradle);

    removeTextureAct = new QAction("Remove &Texture", this);
    connect(removeTextureAct, &QAction::triggered, this, &MainWindow::onRemoveTexture);

    helpAct = new QAction("&Help", this);
    connect(helpAct, &QAction::triggered, this, &MainWindow::onHelp);

    aboutAct = new QAction("&About", this);
    connect(aboutAct, &QAction::triggered, this, &MainWindow::onAbout);

    // Additional actions can be created here
}

void MainWindow::createMenus()
{
    fileMenu = menuBar()->addMenu("&File");
    fileMenu->addAction(openAct);
    fileMenu->addAction(saveAct);
    fileMenu->addAction(saveAsAct);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAct);

    editMenu = menuBar()->addMenu("&Edit");
    editMenu->addAction(detectCradleAct);
    editMenu->addAction(removeCradleAct);
    editMenu->addAction(removeTextureAct);

    helpMenu = menuBar()->addMenu("&Help");
    helpMenu->addAction(helpAct);
    helpMenu->addAction(aboutAct);
}

void MainWindow::createToolbar()
{
    QToolBar *toolbar = addToolBar("Main Toolbar");
    toolbar->addAction(openAct);
    toolbar->addAction(saveAct);
    toolbar->addAction(detectCradleAct);
    toolbar->addAction(removeCradleAct);
    toolbar->addAction(removeTextureAct);

    zoomSlider = new QSlider(Qt::Horizontal);
    zoomSlider->setRange(1, 200);
    zoomSlider->setValue(100);
    connect(zoomSlider, &QSlider::valueChanged, this, &MainWindow::onSlider);

    toolbar->addWidget(zoomSlider);
}

void MainWindow::createStatusBar()
{
    statusLabel = new QLabel("Ready");
    statusBar()->addWidget(statusLabel);
}

void MainWindow::loadImage(const QString &fileName)
{
    if (!imageWidget->loadImage(fileName))
    {
        QMessageBox::warning(this, "Error", "Could not load image.");
        return;
    }

    currentFile = fileName;
    statusLabel->setText("Image loaded: " + fileName);
    updateMenus();
}

void MainWindow::saveImage(const QString &fileName)
{
    if (!imageWidget->saveImage(fileName))
    {
        QMessageBox::warning(this, "Error", "Could not save image.");
        return;
    }

    currentFile = fileName;
    statusLabel->setText("Image saved: " + fileName);
}

void MainWindow::onOpen()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                        "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)");
    debug_fun(fileName, __LINE__);
    if (!fileName.isEmpty())
    {
        loadImage(fileName);
    }
}

void MainWindow::onSave()
{
    if (currentFile.isEmpty())
    {
        onSaveAs();
    }
    else
    {
        saveImage(currentFile);
    }
}

void MainWindow::onSaveAs()
{
    QString fileName = QFileDialog::getSaveFileName(this,
                        "Save Image As", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)");
    if (!fileName.isEmpty())
    {
        saveImage(fileName);
    }
}

void MainWindow::onExit()
{
    close();
}

void MainWindow::onImageChanged()
{
    // Handle when the image changes
    statusLabel->setText("Image updated.");
    updateMenus();
}

void MainWindow::updateMenus()
{
    bool hasImage = imageWidget->hasImage();
    saveAct->setEnabled(hasImage);
    saveAsAct->setEnabled(hasImage);
    detectCradleAct->setEnabled(hasImage);
    removeCradleAct->setEnabled(hasImage);
    removeTextureAct->setEnabled(hasImage);
}

void MainWindow::updateUndo()
{
    // Implement undo functionality if needed
}

void MainWindow::updateSliders()
{
    // Update sliders or other UI elements as needed
}

void MainWindow::onTool()
{
    // Handle tool actions
}

void MainWindow::onToggleOverlay()
{
    // Toggle overlay display if applicable
}

void MainWindow::onToggleResult()
{
    // Toggle result display if applicable
}

void MainWindow::onUndoRedoChanged(bool canUndo)
{
    // Update UI elements based on whether undo/redo is possible
}

void MainWindow::onUndoRedoText(const QString &text)
{
    // Update status or UI with undo/redo text
}

void MainWindow::onCommand()
{
    // Handle commands
}

void MainWindow::onClose()
{
    // Handle commands
}

void MainWindow::onOpenImage()
{
    // Handle commands
}

void MainWindow::onExport()
{
    // Handle commands
}

void MainWindow::onDelete()
{
    // Handle delete action
}

void MainWindow::onTab(int index)
{
    // Handle tab changes
}

void MainWindow::onSelection()
{
    // Handle selection changes
}

void MainWindow::onToolChanged()
{
    // Handle tool changes
}

void MainWindow::onDetectCradle()
{
    if (!imageWidget->hasImage())
    {
        QMessageBox::warning(this, "Error", "No image loaded.");
        return;
    }

    imageWidget->detectCradles();
}

void MainWindow::onRemoveCradle()
{
    if (!imageWidget->hasImage())
    {
        QMessageBox::warning(this, "Error", "No image loaded.");
        return;
    }

    imageWidget->removeCradles();
}

void MainWindow::onRemoveTexture()
{
    if (!imageWidget->hasImage())
    {
        QMessageBox::warning(this, "Error", "No image loaded.");
        return;
    }

    imageWidget->removeTexture();
}

void MainWindow::beginProgress(const QString &msg)
{
    progressDialog = new QProgressDialog(msg, "Cancel", 0, 0, this);
    progressDialog->setWindowModality(Qt::WindowModal);
    connect(progressDialog, &QProgressDialog::canceled, this, &MainWindow::onCancelProgress);
    progressDialog->show();
}

void MainWindow::endProgress()
{
    if (progressDialog)
    {
        progressDialog->close();
        delete progressDialog;
        progressDialog = nullptr;
    }
}

void MainWindow::onProcessingStarted(const QString &message)
{
    beginProgress(message);
}

void MainWindow::onProcessingFinished()
{
    endProgress();
    statusLabel->setText("Processing completed.");
}

void MainWindow::onProcessingCanceled()
{
    endProgress();
    statusLabel->setText("Processing canceled.");
}

void MainWindow::onCancelProgress()
{
    // Handle cancellation
    imageWidget->cancelProcessing();
    statusLabel->setText("Operation canceled.");
}

void MainWindow::onReset()
{
    // Reset the application state
}

void MainWindow::onDone()
{
    // Handle when a processing operation is done
}

void MainWindow::onCancel()
{
    // Handle cancellation
}

void MainWindow::onPolygonAdded(Polygon *poly)
{
    // Handle when a polygon is added
    // For example, add the polygon to the scene or keep track of it
    // If polygons are managed by ImageWidget, you might not need to handle this here
}

void MainWindow::onPolygonRemoved(Polygon *poly)
{
    // Handle when a polygon is removed
    // Remove it from tracking or the scene
}

void MainWindow::onPolygonValueChanged(Polygon *poly, const QString &value)
{
    // Handle when a polygon's value changes
    // Update properties or refresh display as needed
}

void MainWindow::onSlider(int value)
{
    imageWidget->setZoom(value / 100.0);
}

void MainWindow::onBeginChange()
{
    // Handle beginning of a change operation
}

void MainWindow::onEndChange()
{
    // Handle end of a change operation
}

void MainWindow::onHelp()
{
    QMessageBox::information(this, "Help", "This is a tool for removing cradles from X-ray images of art.");
}

void MainWindow::onAbout()
{
    QMessageBox::about(this, "About", "Art Restoration Tool\nVersion 1.0");
}

// Implement other functions as needed
