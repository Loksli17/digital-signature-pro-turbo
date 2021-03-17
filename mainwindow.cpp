#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QImage>
#include <QString>
#include <QMessageBox>


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

    if(this->imagePath != ""){
        this->image.load(this->imagePath);
        ui->imageWrap->setPixmap(this->image);
        QMessageBox::
        QMessageBox::information(this, "Success", "File: " + fileName + " was opened");
    }else{
        QMessageBox::warning(this, "Error", "File wasn't opened");
        return;
    }
}
