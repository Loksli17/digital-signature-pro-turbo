#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QImage>
#include <QString>
#include <QMessageBox>
#include <QDebug>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

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
    QFile file(this->imagePath);


    QStringList lst      = this->imagePath.split('/');
    QString     fileName = lst[lst.count() - 1];

    // it just works
    Mat image = imread("D:\\repos\\digital-signature-pro-turbo\\test.jpg", IMREAD_COLOR);
    QImage img = QImage((uchar*)image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);


    if(this->imagePath != ""){
        this->imagePixels = QPixmap::fromImage(img);
//        this->imagePixels.load(this->imagePath);
        ui->imageWrap->setPixmap(this->imagePixels);
        QMessageBox::information(this, "Success", "File: " + fileName + " was opened");
    }else{
        QMessageBox::warning(this, "Error", "File wasn't opened");
        return;
    }
}
