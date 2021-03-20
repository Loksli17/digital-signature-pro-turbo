#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <stdio.h>
#include <QFileDialog>
#include <QImage>
#include <QString>
#include <QMessageBox>
#include <QDebug>
#include <bitset>
#include <Windows.h>


#include "opencvhelpers.h"

//using namespace cv;
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

void MainWindow::setQualityInfo(int md, double ad, double nad, double mse, double nmse, double snr, double psnr, double IF){
    qDebug() << md << nad;
    ui->md->setText(QString::number(md));
    ui->ad->setText(QString::number(ad));
    ui->nad->setText(QString::number(nad));
    ui->mse->setText(QString::number(mse));
    ui->nmse->setText(QString::number(nmse));
    ui->snr->setText(QString::number(snr));
    ui->psnr->setText(QString::number(psnr));
    ui->IF->setText(QString::number(IF));
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

    string filename = this->imagePath.toStdString();

    string separ(".");
    string::size_type pos = filename.find(separ); // Позиция первого символа строки-разделителя.
    this->first  = filename.substr(0, pos); // Строка до разделителя.
    this->second = filename.substr(pos + separ.length()); // Строка после разделителя


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
    cv::Mat imag = QPixmapToCvMat(this->imagePixels);
    string text = ui->signature->text().toStdString();

    double P = 10; //шаг квантования
//	cout << "Введите порог разности" << endl;/
//	cin >> P;

    int channels = imag.channels();
    cv::Mat image;
    cv::Mat Matvector[3];
    imag.convertTo(imag, CV_32F, 1.0, 0.0);
    cv::Mat charimage;
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

    int N = 4;//размер блока
    vector <cv::Mat> L1, L2, L3;
    vector<bitset<8>> B1;

    int length = text.length();
    for (int i = 0; i < length; i++)
    {
        uchar temp = (uchar)text[i];
        bitset<8>p((temp));
        B1.push_back(p);
    }

    int CVZsize = ceil(sqrt(length * 8));
    cv::Mat CVZ(CVZsize, CVZsize, CV_8U);
    for (int i = 0; i < CVZsize; i++)
    {
        for (int j = 0; j < CVZsize; j++)
        {
            if (i*CVZsize + j < length * 8)
            {
                if (B1[(i*CVZsize + j) / 8][(i*CVZsize + j) % 8] == 0)
                {
                    CVZ.at<uchar>(i, j) = 0;
                }
                if (B1[(i*CVZsize + j) / 8][(i*CVZsize + j) % 8] == 1)
                {
                    CVZ.at<uchar>(i, j) = 255;
                }

            }

        }
    }

    imwrite("CVZ.jpg", CVZ);
    cv::namedWindow(" ЦВЗ", cv::WINDOW_AUTOSIZE);
    imshow(" ЦВЗ", CVZ);
    cv::waitKey(0);
    cv::destroyWindow("ЦВЗ");

    LARGE_INTEGER frequency;
    LARGE_INTEGER t1, t2,t3,t4;
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);
    L1 = WaveletDec();
    int N1 = L1[0].rows / N;
    int N2 = L1[0].cols / N;
    if (8 * text.size() > N1*N2)
    {
//		cout << "Изображение слишком мало для встраивания" << endl;
        int fg;
//		cin >> fg;
    }

    int x1 = 2;
    int y1 = 3;
    int x2 = 3;
    int y2 = 2;
    double k1, k2;
    int it = 0;
    for (int i = 0; i < N1; i++)
    {
        if (i > 0)
        {
            y1 += N;
            y2 += N;
            x1 = 1;
            x2 = 0;
        }
        for (int j = 0; j < N2; j++)
        {
            it++;
            if (it >= text.size() * 8)
            {
                break;
            }
            if (j > 0)
            {
                x1 += N;
                x2 += N;
            }

            k1 = fabs(L1[0].at<float>(y1, x1));
            k2 = fabs(L1[0].at<float>(y2, x2));
            double z1;
            double z2;
            if (L1[0].at<float>(y1, x1) >= 0)
            {
                z1 = 1;
            }
            if (L1[0].at<float>(y1, x1) < 0)
            {
                z1 = -1;
            }
            if (L1[0].at<float>(y2, x2) >= 0)
            {
                z2 = 1;
            }
            if (L1[0].at<float>(y2, x2) < 0)
            {
                z2 = -1;
            }
            if ((B1[(i*N2 + j) / 8][(i*N2 + j) % 8] == 0) && (k1 - k2 <= P))
            {
                k1 = P / 2 + k2 + 1;
                k2 -= P / 2;
            }
            if ((B1[(i*N2 + j) / 8][(i*N2 + j) % 8] == 1) && (k1 - k2 >= -P))
            {
                k2 = P / 2 + k1 + 1;
                k1 -= P / 2;
            }
            L1[0].at<float>(y1, x1) = z1*k1;
            L1[0].at<float>(y2, x2) = z2*k2;
        }
    }

    vector <cv::Mat>LR1;
    cv::Mat imr;
    LR1 = L1;
    imr = WaveletRec(LR1, rows, cols);
    QueryPerformanceCounter(&t2);
    elapsedTime = (float)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;

    /*t1 = clock() - t1;*/
//	cout << "Время встраивания ЦВЗ: " << elapsedTime<< " секунд" << endl;
    cv::Mat imrs;
    imr.convertTo(imrs, CV_8U);
    cv::Mat FResult(rows, cols, CV_32FC3);
    string merged = this->first + "Proposed." + this->second;
    if (channels == 1)
    {
        cv::namedWindow("Wavelet Reconstruction", 1);
        imshow("Wavelet Reconstruction", imrs);
        cv::waitKey(0);
        FResult = imr;
        imwrite(merged, FResult);
    }
    if (channels == 3)
    {
        vector<cv::Mat>Vec;
        Vec.push_back(imr);
        Vec.push_back(Matvector[1]);
        Vec.push_back(Matvector[2]);
        merge(Vec, FResult);
        cv::Mat Fresult1;
        FResult.convertTo(Fresult1, CV_8UC3);
        cv::namedWindow("Wavelet Reconstruction", 1);
        imshow("Wavelet Reconstruction", Fresult1);
        cv::waitKey(0);
        imwrite(merged, Fresult1);
    }


    //проверка качества
    int md = MD(charimage, imrs);
    double ad = AD(charimage, imrs);
    double nad = NAD(charimage, imrs);
    double mse = MSE(charimage, imrs);
    double nmse = NMSE(charimage, imrs);
    double snr = SNR(charimage, imrs);
    double psnr = PSNR(charimage, imrs);
    double If = IF(charimage, imrs);

    qDebug() << "ADASDASDASDASd";
    setQualityInfo(md, ad, nad, mse, nmse, snr, psnr, If);

    string gettext;
    vector<bitset<8>> B2;
    //вейвлет-разложение
    /*clock_t t2 = clock();*/
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);
    vector <cv::Mat> LI1;
    LI1 = WaveletDec();
    //извлечение ЦВЗ
    for (int i = 0; i < length; i++)
    {
        bitset<8>t1;
        for (int j = 0; j < 8; j++)
        {
            t1[j] = 0;
        }
        B2.push_back(t1);
    }

    ////коэффициенты
    x1 = 2;
    y1 = 3;
    x2 = 3;
    y2 = 2;
    it = 0;
    N1 = LI1[0].rows / N;
    N2 = LI1[0].cols / N;
    for (int i = 0; i < N1; i++)
    {
        if ((y1 >= LI1[0].rows) || (y2 >= LI1[0].rows))
        {
            break;
        }
        if (i > 0)
        {
            y1 += N;
            y2 += N;
            x1 = 1;
            x2 = 0;
        }
        for (int j = 0; j < N2; j++)
        {
            it++;

            if (j > 0)
            {
                x1 += N;
                x2 += N;
            }
            if (it >= text.size() * 8)
            {
                break;
            }
            if ((x1 >= LI1[0].cols) || (x2 >= LI1[0].cols))
            {
                break;
            }

            k1 = fabs(LI1[0].at<float>(y1, x1));
            k2 = fabs(LI1[0].at<float>(y2, x2));
            double z1;
            double z2;
            if (LI1[0].at<float>(y1, x1) >= 0)
            {
                z1 = 1;
            }
            if (LI1[0].at<float>(y1, x1) < 0)
            {
                z1 = -1;
            }
            if (LI1[0].at<float>(y2, x2) >= 0)
            {
                z2 = 1;
            }
            if (LI1[0].at<float>(y2, x2) < 0)
            {
                z2 = -1;
            }
            if (k1 - k2 > P)
            {
                B2[(i*N2 + j) / 8][(i*N2 + j) % 8] = 0;
            }
            if (k1 - k2 < -P)
            {
                B2[(i*N2 + j) / 8][(i*N2 + j) % 8] = 1;
            }
        }
    }
    /*t2 = clock() - t2;*/
    QueryPerformanceCounter(&t2);
    elapsedTime = (float)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
//	cout << "Время извлечения ЦВЗ: " << elapsedTime << " секунд" << endl;
    uchar s;
    for (int i = 0; i < text.size(); i++)
    {
        s = B2[i].to_ulong();
        gettext.push_back(s);
    }
//	cout << gettext << endl;
}





// ====ADDITIONAL FUNCTIONS=======
vector<cv::Mat> MainWindow::WaveletDec()
{

    cv::Mat image = QPixmapToCvMat(this->imagePixels);
    cv::Mat im1, im2, im3, im4, im5, im6, imd;
    int rcnt, ccnt;
    float a, b, c, d;

    im1 = cv::Mat::zeros((image.rows+1) / 2, image.cols, CV_32F);
    im2 = cv::Mat::zeros((image.rows+1) / 2, image.cols, CV_32F);
    im3 = cv::Mat::zeros((image.rows+1) / 2, (image.cols+1) / 2, CV_32F);
    im4 = cv::Mat::zeros((image.rows+1) / 2, (image.cols+1) / 2, CV_32F);
    im5 = cv::Mat::zeros((image.rows+1) / 2, (image.cols+1) / 2, CV_32F);
    im6 = cv::Mat::zeros((image.rows+1) / 2, (image.cols+1) / 2, CV_32F);

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


    vector<cv::Mat> result;
    result.push_back(im3.clone());
    result.push_back(im4.clone());
    result.push_back(im5.clone());
    result.push_back(im6.clone());
    return result;
}

cv::Mat MainWindow::WaveletRec(vector<cv::Mat> Decomp,int rows,int cols)
{
    float a, b, c, d;
    int ccnt, rcnt;
    cv::Mat result = cv::Mat::zeros(rows, cols, CV_32F);
    cv::Mat im11   = cv::Mat::zeros((rows+1) / 2, cols, CV_32F);
    cv::Mat im12   = cv::Mat::zeros((rows+1) / 2, cols, CV_32F);
    cv::Mat im13   = cv::Mat::zeros((rows+1) / 2, cols, CV_32F);
    cv::Mat im14   = cv::Mat::zeros((rows+1) / 2, cols, CV_32F);

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

    cv::Mat  temp = cv::Mat::zeros(rows, cols, CV_32F);
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

vector<cv::Mat> MainWindow::WaveletDec8()
{
    cv::Mat image = QPixmapToCvMat(this->imagePixels);

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

    cv::Mat Wave;
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
    vector<cv::Mat> result;
    cv::Mat im3(rows / 2, cols / 2, CV_32F);
    cv::Mat im4(rows / 2, cols / 2, CV_32F);
    cv::Mat im5(rows / 2, cols / 2, CV_32F);
    cv::Mat im6(rows / 2, cols / 2, CV_32F);

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

int MainWindow::MD(cv::Mat cont, cv::Mat stego)//максимальная разность
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

double MainWindow::AD(cv::Mat cont, cv::Mat stego)//средняя абсолютная разность
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

double MainWindow::NAD(cv::Mat cont, cv::Mat stego)//нормированная средняя абсолютная разность
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

double MainWindow::MSE(cv::Mat cont, cv::Mat stego)//среднеквадратическая ошибка
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

double MainWindow::NMSE(cv::Mat cont, cv::Mat stego)//нормированная среднеквадратическая ошибка
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

double MainWindow::SNR(cv::Mat cont, cv::Mat stego)//отношение "сигнал-шум"
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

double MainWindow::PSNR(cv::Mat cont, cv::Mat stego)//максимальное отношение "сигнал-шум"
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

double MainWindow::IF(cv::Mat cont, cv::Mat stego)//качество изображения
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
