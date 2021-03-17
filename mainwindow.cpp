#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QImage>
#include <QString>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_loadImage_clicked()
{
    ui->imageWrap->clear();

    this->imagePath = QFileDialog::getOpenFileName(nullptr, "Choose image", "C:/desktop", "*.jpg");

    if(this->imagePath != ""){
        this->image.load(this->imagePath);
        ui->imageWrap->setPixmap(this->image);
    }
}
