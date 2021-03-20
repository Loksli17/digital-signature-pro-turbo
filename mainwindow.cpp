#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <stdio.h>
#include <QFileDialog>
#include <QImage>
#include <QString>
#include <QMessageBox>
#include <QDebug>
#include <bitset>


#include "opencvhelpers.h"

using namespace cv;
using namespace std;


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


//ui->imageWrap->setPixmap(cvMatToQPixmap(mat));
//QPixmapToCvMat(this->imagePixels);

void MainWindow::on_loadImage_clicked()
{

    //TODO вынести код потом отдельно, потому что он будет выполняться с других мест
    ui->imageWrap->clear();

    this->imagePath = QFileDialog::getOpenFileName(nullptr, "Choose image", "C:/desktop", "*.jpg");

    QFile       file(this->imagePath);
    QStringList lst      = this->imagePath.split('/');
    QString     fileName = lst[lst.count() - 1];

    if(this->imagePath != ""){
        this->imagePixels.load(this->imagePath);
        ui->imageWrap->setPixmap(this->imagePixels);
        setWindowTitle(fileName);
        QMessageBox::information(this, "Success", "File: " + fileName + " was opened");
    }else{
        QMessageBox::warning(this, "Error", "File wasn't opened");
        return;
    }
}

void MainWindow::on_authorAlgorithm_clicked()
{
    Mat imag = QPixmapToCvMat(this->imagePixels);

    double P; //шаг квантования
//	cout << "Введите порог разности" << endl;/
//	cin >> P;

    int channels = imag.channels();
    Mat image;
    Mat Matvector[3];
    imag.convertTo(imag, CV_32F, 1.0, 0.0);
    Mat charimage;
    if (channels == 1) //чёрно-белое изображение
    {
        image = imag;
        image.convertTo(charimage, CV_8U);
    }
    if (channels == 3) //цветное изображение
    {

        split(imag, Matvector);
        image = Matvector[0];//встраивание в синюю компоненту
        image.convertTo(charimage, CV_8U);
    }
    int rows = image.rows;
    int cols = image.cols;
}





// ====ADDITIONAL FUNCTIONS=======
vector<Mat> MainWindow::WaveletDec()
{

    Mat image = QPixmapToCvMat(this->imagePixels);
    Mat im1, im2, im3, im4, im5, im6, imd;
    int rcnt, ccnt;
    float a, b, c, d;

    im1 = Mat::zeros((image.rows+1) / 2, image.cols, CV_32F);
    im2 = Mat::zeros((image.rows+1) / 2, image.cols, CV_32F);
    im3 = Mat::zeros((image.rows+1) / 2, (image.cols+1) / 2, CV_32F);
    im4 = Mat::zeros((image.rows+1) / 2, (image.cols+1) / 2, CV_32F);
    im5 = Mat::zeros((image.rows+1) / 2, (image.cols+1) / 2, CV_32F);
    im6 = Mat::zeros((image.rows+1) / 2, (image.cols+1) / 2, CV_32F);

    for (rcnt = 0; rcnt<image.rows; rcnt += 2)
    {
        for (ccnt = 0; ccnt<image.cols; ccnt++)
        {

            a = image.at<float>(rcnt, ccnt);
            if (rcnt + 1 >= image.rows)
            {
                b = image.at<float>(rcnt + 1- image.rows,ccnt);
            }
            if (rcnt + 1 < image.rows)
            {
                b = image.at<float>(rcnt + 1, ccnt);
            }
            c = (a + b)*0.707;
            d = (a - b)*0.707;
            int _rcnt = rcnt / 2;
            im1.at<float>(_rcnt, ccnt) = c;
            im2.at<float>(_rcnt, ccnt) = d;
        }
    }

    for (rcnt = 0; rcnt<(image.rows+1) / 2; rcnt++)
    {
        for (ccnt = 0; ccnt<image.cols; ccnt += 2)
        {

            a = im1.at<float>(rcnt, ccnt);
            if (ccnt + 1 >= image.cols)
            {
                b = im1.at<float>(rcnt, ccnt + 1 - im1.cols);
            }
            if (ccnt + 1 < image.cols)
            {
                b = im1.at<float>(rcnt, ccnt + 1);
            }
            c = (a + b)*0.707;
            d = (a - b)*0.707;
            int _ccnt = ccnt / 2;
            im3.at<float>(rcnt, _ccnt) = c;
            im4.at<float>(rcnt, _ccnt) = d;
        }
    }

    for (rcnt = 0; rcnt<(image.rows+1)/ 2; rcnt++)
    {
        for (ccnt = 0; ccnt<image.cols; ccnt += 2)
        {

            a = im2.at<float>(rcnt, ccnt);
            if (ccnt + 1 >= image.cols)
            {
                b = im2.at<float>(rcnt, ccnt + 1 - im1.cols);
            }
            if (ccnt + 1 < image.cols)
            {
                b = im2.at<float>(rcnt, ccnt + 1);
            }

            c = (a + b)*0.707;
            d = (a - b)*0.707;
            int _ccnt = ccnt / 2;
            im5.at<float>(rcnt, _ccnt) = c;
            im6.at<float>(rcnt, _ccnt) = d;
        }
    }


    vector<Mat> result;
    result.push_back(im3.clone());
    result.push_back(im4.clone());
    result.push_back(im5.clone());
    result.push_back(im6.clone());
    return result;
}

Mat MainWindow::WaveletRec(vector<Mat> Decomp,int rows,int cols)
{
    float a, b, c, d;
    int ccnt, rcnt;
    Mat result = Mat::zeros(rows, cols, CV_32F);
    Mat im11 = Mat::zeros((rows+1) / 2, cols, CV_32F);
    Mat im12 = Mat::zeros((rows+1) / 2, cols, CV_32F);
    Mat im13 = Mat::zeros((rows+1) / 2, cols, CV_32F);
    Mat im14 = Mat::zeros((rows+1) / 2, cols, CV_32F);

    for (rcnt = 0; rcnt<(rows+1) / 2; rcnt++)
    {
        for (ccnt = 0; ccnt<(cols+1) / 2; ccnt++)
        {
            int _ccnt = ccnt * 2;
            im11.at<float>(rcnt, _ccnt) = Decomp[0].at <float>(rcnt, ccnt);     //Upsampling of stage I
            im12.at<float>(rcnt, _ccnt) = Decomp[1].at<float>(rcnt, ccnt);
            im13.at<float>(rcnt, _ccnt) = Decomp[2].at<float>(rcnt, ccnt);
            im14.at<float>(rcnt, _ccnt) = Decomp[3].at<float>(rcnt, ccnt);
        }
    }


    for (rcnt = 0; rcnt<(rows+1) / 2; rcnt++)
    {
        for (ccnt = 0; ccnt<cols; ccnt += 2)
        {

            a = im11.at<float>(rcnt, ccnt);
            b = im12.at<float>(rcnt, ccnt);
            c = (a + b)*0.707;
            im11.at<float>(rcnt, ccnt) = c;
            d = (a - b)*0.707;                     //Filtering at Stage I
            if (ccnt + 1 < cols)
            {
                im11.at<float>(rcnt, ccnt + 1) = d;
            }
            if (ccnt + 1 >= cols)
            {
                im11.at<float>(rcnt, ccnt + 1-cols) = d;
            }

            a = im13.at<float>(rcnt, ccnt);
            b = im14.at<float>(rcnt, ccnt);
            c = (a + b)*0.707;
            im13.at<float>(rcnt, ccnt) = c;
            d = (a - b)*0.707;
            if (ccnt + 1 < cols)
            {
                im13.at<float>(rcnt, ccnt + 1) = d;
            }
            if (ccnt + 1 >= cols)
            {
                im13.at<float>(rcnt, ccnt + 1 - cols) = d;
            }
        }
    }

    Mat  temp = Mat::zeros(rows, cols, CV_32F);
    for (rcnt = 0; rcnt<(rows+1) / 2; rcnt++)
    {
        for (ccnt = 0; ccnt<cols; ccnt++)
        {

            int _rcnt = rcnt * 2;
            result.at<float>(_rcnt, ccnt) = im11.at<float>(rcnt, ccnt);     //Upsampling at stage II
            temp.at<float>(_rcnt, ccnt) = im13.at<float>(rcnt, ccnt);
        }
    }

    for (rcnt = 0; rcnt<rows; rcnt += 2)
    {
        for (ccnt = 0; ccnt<cols; ccnt++)
        {

            a = result.at<float>(rcnt, ccnt);
            b = temp.at<float>(rcnt, ccnt);
            c = (a + b)*0.707;
            result.at<float>(rcnt, ccnt) = c;                                      //Filtering at Stage II
            d = (a - b)*0.707;
            if (rcnt+1 < rows)
            {
                result.at<float>(rcnt + 1, ccnt) = d;
            }
            if (rcnt+1 >= rows)
            {
                result.at<float>(rcnt + 1 - rows, ccnt) = d;
            }

        }
    }
    return result;
}

vector<Mat> MainWindow::WaveletDec8()
{
    Mat image = QPixmapToCvMat(this->imagePixels);

    float c = 0.9;
    int rows = image.rows;
    int cols = image.cols;
    int size = rows*cols;
    float c0, c1, c2, c3, c4, c5, c6, c7;

    c0 = 0.3258;
    c1 = 1.0109;
    c2 = 0.8922;
    c3 = -0.0396;
    c4 = -0.2645;
    c5 = 0.04636;
    c6 = 0.0465;
    c7 = -0.015;

    Mat Wave;
    Wave = image.reshape(1,1);
    image = image.reshape(1, 1);
    float sum1;
    float sum2;

    for (int i = 0; i < size; i=i+2)
    {
            sum1 = 0;
            sum2 = 0;
            sum1 = image.at<float>(i)*c0+image.at<float>(i+1)*c1;
            sum2 = image.at<float>(i)*c7-c6*image.at<float>(i+1);
            if (i + 2 < size)
            {
                sum1 += image.at<float>(i + 2)*c2 + image.at<float>(i + 3)*c3;
                sum2 += image.at<float>(i + 2)*c5 - c4* image.at<float>(i + 3);
            }
            if (i + 2 >= size)
            {
                sum1 += image.at<float>(i + 2-size)*c2 + image.at<float>(i + 3-size)*c3;
                sum2 += image.at<float>(i + 2-size)*c5 - c4* image.at<float>(i + 3-size);
            }
            if (i + 4 < size)
            {
                sum1 += image.at<float>(i + 4)*c4 + image.at<float>(i + 5)*c5;
                sum2 += image.at<float>(i + 4)*c3 - c2* image.at<float>(i + 5);
            }
            if (i + 4>= size)
            {
                sum1 += image.at<float>(i + 4-size)*c4 + image.at<float>(i + 5-size)*c3;
                sum2 += image.at<float>(i + 4-size)*c3 - c2* image.at<float>(i + 5-size);
            }
            if (i + 6 < size)
            {
                sum1 += image.at<float>(i + 6)*c6 + image.at<float>(i + 7)*c7;
                sum2 += image.at<float>(i + 6)*c1 - c0*image.at<float>(i + 7);
            }
            if (i + 6 >= size)
            {
                sum1 += image.at<float>(i + 6-size)*c6 + image.at<float>(i + 7-size)*c7;
                sum2 += image.at<float>(i + 6-size)*c1 - c0*image.at<float>(i + 7-size);
            }

            Wave.at<float>(i) = sum1*c;
            Wave.at<float>(i + 1) = sum2*c;
    }

    Wave = Wave.reshape(1,rows);
    vector<Mat> result;
    Mat im3(rows / 2, cols / 2, CV_32F);
    Mat im4(rows / 2, cols / 2, CV_32F);
    Mat im5(rows / 2, cols / 2, CV_32F);
    Mat im6(rows / 2, cols / 2, CV_32F);

    for (int i = 0; i < rows / 2; i++)
    {
        for (int j = 0; j < cols / 2; j++)
        {
            im3.at<float>(i, j) = Wave.at<float>(i, j);
        }
    }

    for (int i = 0; i < rows / 2; i++)
    {
        for (int j = cols / 2; j < cols; j++)
        {
            im4.at<float>(i, j-cols/2) = Wave.at<float>(i,j);
        }
    }

    for (int i = rows / 2; i < rows; i++)
    {
        for (int j = 0; j < cols / 2; j++)
        {
            im5.at<float>(i-rows/2, j) = Wave.at<float>(i, j);
        }
    }

    for (int i = rows / 2; i < rows; i++)
    {
        for (int j = cols / 2; j < cols; j++)
        {
            im6.at<float>(i-rows/2, j-cols/2) = Wave.at<float>(i, j);
        }
    }

    result.push_back(im3.clone());
    result.push_back(im4.clone());
    result.push_back(im5.clone());
    result.push_back(im6.clone());
    return result;
}


double MainWindow::sigma1(int x) //вспомогательная функция для алгоритма Коха
{
    double sigma;
    if (x == 0)
    {
        sigma = 1 / sqrt(2);
    }
    if (x > 0)
    {
        sigma = 1;
    }
    return sigma;

}

int MainWindow::MD(Mat cont, Mat stego)//максимальная разность
{
    int max = abs(cont.at<uchar>(0, 0) - stego.at<uchar>(0, 0));
    for (int i = 0; i < cont.rows; i++)
    {
        for (int j = 0; j < cont.cols; j++)
        {
            int md = abs(cont.at<uchar>(i, j) - stego.at<uchar>(i, j));
            if (md>max)
            {
                max = md;
            }
        }
    }
    return max;

}

double MainWindow::AD(Mat cont, Mat stego)//средняя абсолютная разность
{

    double sum = 0;
    for (int i = 0; i < cont.rows; i++)
    {
        for (int j = 0; j < cont.cols; j++)
        {
            sum += abs(cont.at<uchar>(i, j) - stego.at<uchar>(i, j));
        }
    }
    double ad = sum / (cont.rows*cont.cols);
    return ad;

}

double MainWindow::NAD(Mat cont, Mat stego)//нормированная средняя абсолютная разность
{
    double sum1 = 0;
    double sum2 = 0;
    for (int i = 0; i < cont.rows; i++)
    {
        for (int j = 0; j < cont.cols; j++)
        {
            sum1 += abs(cont.at<uchar>(i, j) - stego.at<uchar>(i, j));
            sum2 += cont.at<uchar>(i, j);
        }
    }
    double nad = sum1 / sum2;
    return nad;
}

double MainWindow::MSE(Mat cont, Mat stego)//среднеквадратическая ошибка
{
    double sum = 0;
    for (int i = 0; i < cont.rows; i++)
    {
        for (int j = 0; j < cont.cols; j++)
        {
            sum += pow(cont.at<uchar>(i, j) - stego.at<uchar>(i, j), 2);
        }
    }
    double mse = sum / (cont.rows*cont.cols);
    return mse;
}

double MainWindow::NMSE(Mat cont, Mat stego)//нормированная среднеквадратическая ошибка
{
    double sum1 = 0;
    double sum2 = 0;
    for (int i = 0; i < cont.rows; i++)
    {
        for (int j = 0; j < cont.cols; j++)
        {
            sum1 += pow(cont.at<uchar>(i, j) - stego.at<uchar>(i, j), 2);
            sum2 += pow(cont.at<uchar>(i, j), 2);
        }
    }
    double nmse = sum1 / sum2;
    return nmse;
}

double MainWindow::SNR(Mat cont, Mat stego)//отношение "сигнал-шум"
{
    double sum1 = 0;
    double sum2 = 0;
    for (int i = 0; i < cont.rows; i++)
    {
        for (int j = 0; j < cont.cols; j++)
        {
            sum1 += pow(cont.at<uchar>(i, j) - stego.at<uchar>(i, j), 2);
            sum2 += pow(cont.at<uchar>(i, j), 2);
        }
    }
    double snr = sum2 / sum1;
    return snr;
}

double MainWindow::PSNR(Mat cont, Mat stego)//максимальное отношение "сигнал-шум"
{
    double sum1 = 0;
    int max = cont.at<uchar>(0, 0);
    for (int i = 0; i < cont.rows; i++)
    {
        for (int j = 0; j < cont.cols; j++)
        {
            sum1 += pow(cont.at<uchar>(i, j) - stego.at<uchar>(i, j), 2);
            int ps = cont.at<uchar>(i, j);
            if (max < ps)
            {
                max = ps;
            }
        }
    }

    double psnr = cont.rows*cont.cols * pow(max, 2) / sum1;
    return psnr;
}

double MainWindow::IF(Mat cont, Mat stego)//качество изображения
{
    double sum1 = 0;
    double sum2 = 0;
    for (int i = 0; i < cont.rows; i++)
    {
        for (int j = 0; j < cont.cols; j++)
        {
            sum1 += pow(cont.at<uchar>(i, j) - stego.at<uchar>(i, j), 2);
            sum2 += pow(cont.at<uchar>(i, j), 2);
        }
    }
    double if1 = 1 - (sum1 / sum2);
    return if1;
}
