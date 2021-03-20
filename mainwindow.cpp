#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <stdio.h>
#include <QFileDialog>
#include <QImage>
#include <QString>
#include <QMessageBox>
#include <QDebug>


#include "opencvhelpers.h"

//using namespace cv;

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

    //TODO вынести код потом отдельно, потому что он будет выполняться с других мест
    ui->imageWrap->clear();

    this->imagePath = QFileDialog::getOpenFileName(nullptr, "Choose image", "C:/desktop", "*.jpg");

    QFile       file(this->imagePath);
    QStringList lst      = this->imagePath.split('/');
    QString     fileName = lst[lst.count() - 1];

    if(this->imagePath != ""){
//        this->imagePixels = QPixmap::fromImage(img);
        this->imagePixels.load(this->imagePath);

        // It's fast, it's furios, it just works
        // QPixmap to cv::Mat
        mat = QPixmapToCvMat(this->imagePixels);
        // cv::Mat to QPixmap
        ui->imageWrap->setPixmap(cvMatToQPixmap(mat));
        setWindowTitle(fileName);
        QMessageBox::information(this, "Success", "File: " + fileName + " was opened");
    }else{
        QMessageBox::warning(this, "Error", "File wasn't opened");
        return;
    }
}
