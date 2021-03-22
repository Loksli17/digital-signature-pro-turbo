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
#include <QException>


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

void MainWindow::setCurrentAlgorithm(QString name)
{
    currentAlgorithm = name;
    ui->currentAlgorithm->setText(currentAlgorithm);
}

void MainWindow::decode()
{
    if (currentAlgorithm == "koch") {
        decodeKoch();
    }else if(currentAlgorithm == "author"){
        decodeAuthor();
    }else if(currentAlgorithm == "sanghavi"){
        decodeSanghavi();
    } else if(currentAlgorithm == "soheili") {
        decodeSoheili();
    }
}


void MainWindow::decodeKoch()
{
    cv::Mat imag = QPixmapToCvMat(this->imagePixels);
    cv::Mat FResult = QPixmapToCvMat(imageProcessedPixels);

    int contsize = imag.rows * imag.cols;

    int N = 8;//размер блока
    int Nc = contsize / (N * N);
    double summ, znach;

    string text = ui->signature->text().toStdString();
    int length = text.length();

    int heigh = FResult.rows;
    int widt = FResult.cols;
    //извлечение ЦВЗ
//    cout << "Извлечение ЦВЗ" << endl;
    clock_t t2 = clock();
    int Nc1 = widt*heigh / (N * N);
    //разбиение на сегменты
    vector<cv::Mat>coofs1;
    int ce = 0;
    int re = 0;
    for (int i = 0; i < Nc1; i++)
    {
        cv::Mat C(N, N, CV_8UC1);
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                if ((ce + j < heigh) && (re + k < widt))
                {
                    C.at<uchar>(j, k) = Matvector[0].at<uchar>(ce + j, re + k);
                }


            }
        }
        re += N;
        if (re >= widt)
        {
            ce += N;
            re = 0;
        }
        coofs1.push_back(C);
    }

    //дискретное косинусное преобразование
    vector<double**> Sigma1;
    for (int b = 0; b < Nc1; b++)
    {
        double** s = new double*[N];
        for (int m = 0; m < N; m++)
        {
            double* si = new double[N];
            s[m] = si;

        }

        for (int v = 0; v < N; v++)
        {
            for (int v1 = 0; v1 < N; v1++)
            {
                summ = 0;
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        summ += coofs1[b].at<uchar>(i, j)*cos(CV_PI*v * (2 * i + 1) / (2 * N))*cos(CV_PI*v1 * (2 * j + 1) / (2 * N));

                    }
                }
                znach = (sigma1(v)*sigma1(v1) / (sqrt(2 * N)))*summ;
                s[v][v1] = znach;



            }
        }
        Sigma1.push_back(s);
    }

    int it = 0;
    vector<uchar>finalcodes;
    bitset<8>cod;
    for (int k = 0; k < length * 8; k++) {
        om1 = fabs(Sigma1[k][x1][y1]);
        om2 = fabs(Sigma1[k][x2][y2]);

        if (om1>om2)
        {
            cod[it] = 0;
        }

        if (om1 < om2)
        {
            cod[it] = 1;
        }
        if (it < 8)
        {
            it++;
        }
        if (it == 8)
        {
            it = 0;
            finalcodes.push_back(cod.to_ulong());
            bitset<8>cod;
        }

    }

    string ftext;
    for (int i = 0; i < length; i++)
    {
        ftext.push_back(finalcodes[i]);
    }
    t2 = clock() - t2;

    ui->decodedSignature->setText(QString::fromStdString(ftext));
    ui->duration->setText(QString::number(t2 / CLOCKS_PER_SEC) + " sec");
}

void MainWindow::decodeAuthor(){

//    cv::Mat imr     = algResult;
    cv::Mat Fresult = FResult;
    string text = ui->signature->text().toStdString();
    int P = 10;
    int length = text.length();

    if (imr.channels() == 1)
    {
        imr = Fresult;
    }
    if (imr.channels() == 3)
    {
        split(Fresult, Matvector);
        Matvector[0].convertTo(imr, CV_32FC1, 1.0, 0.0);
    }

    string gettext;
    vector<bitset<8>> B2;
    //вейвлет-разложение
    /*clock_t t2 = clock();*/
    QueryPerformanceFrequency(&this->authorFrequency);
    QueryPerformanceCounter(&this->authorT1);
    vector <cv::Mat> LI1;
    LI1 = WaveletDec(imr);

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
    int it = 0;

    int N = this->authorN;
    int N1 = LI1[0].rows / N;
    int N2 = LI1[0].cols / N;

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

            double k1 = fabs(LI1[0].at<float>(y1, x1));
            double k2 = fabs(LI1[0].at<float>(y2, x2));
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
    QueryPerformanceCounter(&this->authorT2);
    double elapsedTime = (float)(this->authorT2.QuadPart - this->authorT1.QuadPart) / this->authorFrequency.QuadPart;
//	cout << "Время извлечения ЦВЗ: " << elapsedTime << " секунд" << endl;
    uchar s;
    for (int i = 0; i < text.size(); i++)
    {
        s = B2[i].to_ulong();
        gettext.push_back(s);
    }

    qDebug() << "popa: " << QString::fromStdString(gettext);

    ui->decodedSignature->setText(QString::fromStdString(gettext));
    ui->duration->setText(QString::number(elapsedTime, 'f', 3) + " sec");
}

void MainWindow::decodeSanghavi(){

//    cv::Mat imr     = algResult;
    cv::Mat Fresult = FResult;
    string text     = ui->signature->text().toStdString();
    int P = 10;
    int length = text.length();

    qDebug() << Fresult.rows;

    if (imr.channels() == 1)
    {
        imr = Fresult;
    }
    if (imr.channels() == 3)
    {
        split(Fresult, Matvector);
        Matvector[0].convertTo(imr, CV_32FC1, 1.0, 0.0);
    }

    string gettext;
    vector<bitset<8>> B2;
    //вейвлет-разложение
    clock_t tt2 = clock();
    vector <cv::Mat> LI1, LI2, LI3, LI4;
    LI1 = WaveletDec(imr);
    LI2 = WaveletDec(LI1[0]);
    LI3 = WaveletDec(LI2[0]);
    int i, j, k;
    //извлечение ЦВЗ
    for (i = 0; i < length; i++)
    {
        bitset<8>t1;
        for (j = 0; j < 8; j++)
        {
            t1[j] = 0;
        }
        B2.push_back(t1);
    }

    int k1, c;
    cv::Mat Array = LI3[2].reshape(1, 1);
    uchar t2;
    for (i = 0; i < length; i++)
    {
        for (j = 0; j < 8; j++)
        {
            c = 0;
            k1 = (i * 8 + j) * 5;
            if (k1+4 >= Array.cols)
            {
                break;
            }
            for (k = 1; k < 5; k++)
            {
                if (Array.at<float>(k1) > Array.at<float>(k1 + k))
                {
                    c++;
                }

            }
            if (c > 2)
            {
                B2[i][j] = 1;
            }
            if (c <= 2)
            {
                B2[i][j] = 0;
            }

        }
        t2 = B2[i].to_ulong();
        gettext.push_back(t2);

    }
    tt2 = clock() - tt2;

    ui->decodedSignature->setText(QString::fromStdString(gettext).toUtf8());
    ui->duration->setText(QString::number(tt2 / CLOCKS_PER_SEC) + " sec");

    qDebug() << "jopa";
}

void MainWindow::decodeSoheili()
{
    int i, j, k;
    int m;
    int channels = this->FResult.channels();
    int rows = this->FResult.rows;
    int cols = this->FResult.cols;

//    string text

    if (channels == 1)
    {
        this->RW = this->FResult;
    }
    if (channels == 3)
    {
        split(this->FResult, Matvector);
        Matvector[0].convertTo(this->RW, CV_32FC1, 1.0, 0.0);
    }
    //обратный ход
    clock_t tt2 = clock();
    vector <cv::Mat> LW1, LW2, LW3;
    LW1 = WaveletDec(this->RW);
    LW2 = WaveletDec(LW1[0]);
    LW2[0] =LW2[0].reshape(1, 1);
    //извлечение ЦВЗ
    vector <bitset<8>> B2;
    for (k = 0; k < this->K; k++)
    {
        for (i = 0; i < this->text.length(); i++)
        {
            bitset<8> temp;
            for (j = 0; j < 8; j++)
            {
                m = LW2[0].at <float>(k * this->wsize + i * 8 + j) / Q;
                if ((LW2[0].at <float>(k * this->wsize + i * 8 + j) > (m + 0.75) * Q) || (LW2[0].at <float>(k * this->wsize + i * 8 + j) <= (m + 0.25) * Q))
                {
                    temp[j] = 1;

                }

                if ((LW2[0].at <float>(k * this->wsize + i * 8 + j) > (m + 0.25) * Q) && (LW2[0].at <float>(k * this->wsize + i * 8 + j) <= (m + 0.75) * Q))
                {
                    temp[j] = 0;
                }
            }
            B2.push_back(temp);
        }
    }
    //анализ результатов
    vector<int> summ (this->text.length() * 8);
    for (k = 0; k < K; k++)
    {
        for (i = 0; i < this->text.length(); i++)
        {
            for (j = 0; j < 8; j++)
            {
                summ[i * 8 + j] += B2[k * this->text.length() + i][j];
            }
        }
    }

    string gettext;
    uchar t;
    for (i = 0; i < this->text.length(); i++)
    {
        bitset<8>temp2;
        for (j = 0; j < 8; j++)
        {
            if (summ[i * 8 + j] >= this->K / 2)
            {
                temp2[j] = 1;
            }
            else
            {
                temp2[j] = 0;
            }

        }
        t = temp2.to_ulong();
        gettext.push_back(t);
    }
    tt2 = clock() - tt2;

    qDebug() << QString::fromStdString(gettext);

    ui->duration->setText(QString::number(tt2 / CLOCKS_PER_SEC) + " secs");
    ui->decodedSignature->setText(QString::fromStdString(gettext));
//    cout << "Время извлечения ЦВЗ: " << (double)tt2 / CLOCKS_PER_SEC << " секунд" << endl;
//    cout << gettext << endl;
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

//        cv::Mat image = QPixmapToCvMat(this->imagePixels);
        cv::Mat image = QtOcv::image2Mat(imagePixels.toImage());
        this->width  = image.rows;
        this->height = image.cols;

        qDebug() << this->width << this->height;

        setWindowTitle(fileName);
        QMessageBox::information(this, "Success", "File: " + fileName + " was opened");
    }else{
        QMessageBox::warning(this, "Error", "File wasn't opened");
        return;
    }
}

void MainWindow::on_authorAlgorithm_clicked()
{
    setCurrentAlgorithm("author");

    cv::Mat imag = QPixmapToCvMat(this->imagePixels);
//    cv::Mat imag = QtOcv::image2Mat(imagePixels.toImage(), CV_8UC3);

    string text = ui->signature->text().toStdString();

    double P = 10; //шаг квантования
//	cout << "Введите порог разности" << endl;/
//	cin >> P;

    int channels = imag.channels();
    cv::Mat image;
//    cv::Mat Matvector[3];
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
    this->authorN = N;

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
                if (B1[(i * CVZsize + j) / 8][(i * CVZsize + j) % 8] == 0)
                {
                    CVZ.at<uchar>(i, j) = 0;
                }
                if (B1[(i * CVZsize + j) / 8][(i * CVZsize + j) % 8] == 1)
                {
                    CVZ.at<uchar>(i, j) = 255;
                }

            }

        }
    }

//    imwrite("CVZ.jpg", CVZ);
//    cv::namedWindow(" ЦВЗ", cv::WINDOW_AUTOSIZE);
//    imshow(" ЦВЗ", CVZ);
//    cv::waitKey(0);
//    cv::destroyWindow("ЦВЗ");

    LARGE_INTEGER frequency;
    LARGE_INTEGER t1, t2, t3, t4;

    this->authorFrequency = frequency;
    this->authorT1        = t1;
    this->authorT2        = t2;


    double elapsedTime;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);
    L1 = WaveletDec(image);
    int N1 = L1[0].rows / N;
    int N2 = L1[0].cols / N;

    if (8 * text.size() > N1 * N2)
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
//    cv::Mat imr;
    LR1 = L1;
    imr = WaveletRec(LR1, rows, cols);
    QueryPerformanceCounter(&t2);
    elapsedTime = (float)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;

    ui->duration->setText(QString::number(elapsedTime, 'f', 3) + " sec");

    cv::Mat imrs;
    imr.convertTo(imrs, CV_8U);
    cv::Mat FResult(rows, cols, CV_32FC3);
    string merged = this->first + "Proposed." + this->second;

    qDebug() << channels;
    qDebug() << "ты чо бля";

    if (channels == 1)
    {
        cv::namedWindow("Wavelet Reconstruction", 1);

        imageProcessedPixels = cvMatToQPixmap(imrs);
        qDebug() << "Kek происходит тут";
//        imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(imrs));
        ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);

        FResult = imr;
        imageProcessedPixels = cvMatToQPixmap(FResult);
        this->algResult = imr;
        this->FResult   = imr;
//        imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult));
//        imwrite(merged, FResult);
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
//        cv::namedWindow("Wavelet Reconstruction", 1);
        qDebug() << "Kek происходит тут";
        imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(Fresult1));
//        imageProcessedPixels = cvMatToQPixmap(Fresult1);
        ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
//        imageProcessedPixels = cvMatToQPixmap(FResult);
//        imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult));
        this->algResult = FResult;
        this->FResult   = FResult;

//        imwrite(merged, Fresult1);
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

    setQualityInfo(md, ad, nad, mse, nmse, snr, psnr, If);
}

void MainWindow::on_kochAlgorithm_clicked()
{
    setCurrentAlgorithm("koch");

    double P = 10; //шаг квантования
    string text = ui->signature->text().toStdString();
    cv::Mat imag = QPixmapToCvMat(this->imagePixels);

    int i, j, k;
    int channels = imag.channels();
    int widt = imag.cols;
    int heigh = imag.rows;
    int contsize = widt*heigh;
    int length = text.length();
    cv::Mat start(heigh,widt,CV_8UC1);

    if (channels == 1) //чёрно-белое изображение
    {
        start = imag;
    }
    if (channels == 3) //цветное изображение
    {
        split(imag, Matvector);
        for (i = 0; i < heigh; i++)
        {
            for (j = 0; j < widt; j++)
            {
                start.at<uchar>(i, j) = Matvector[0].at<uchar>(i, j);
            }
        }
    }
    int N = 8;//размер блока
    int Nc = contsize / (N * N);
    if (8 * text.size() > Nc){
        //        cout << "Изображение слишком мало для встраивания" << endl;
        int fg;
        //        cin >> fg;

    }
    vector<bitset<8>> M;
    uchar temp;
    for (i = 0; i < length; i++)
    {
        temp = (uchar)text[i];
        bitset<8>m((temp));
        M.push_back(m);
    }
    //    cout << "Встраивание ЦВЗ" << endl;
    clock_t t1 = clock();
    //разбиение на сегменты
    vector<cv::Mat>coofs;
    int c = 0;
    int r = 0;
    for (i = 0; i < Nc; i++)
    {
        cv::Mat C(N, N, CV_8UC1);
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                if ((c + j < heigh) && (r + k < widt))
                {
                    C.at<uchar>(j, k) = Matvector[0].at<uchar>(c + j, r + k);
                }
            }
        }
        r += N;
        if (r >= widt)
        {
            c += N;
            r = 0;
        }
        coofs.push_back(C);
    }

    //Вычисление коэффициентов ДКП
    vector<double**> sigma;
    for (int b = 0; b < Nc; b++)
    {
        double** s = new double*[N];
        for (int m = 0; m < N; m++)
        {
            double* si = new double[N];
            s[m] = si;

        }

        for (int v = 0; v < N; v++)
        {
            for (int v1 = 0; v1 < N; v1++)
            {
                double summ = 0;
                for (i = 0; i < N; i++)
                {
                    for (j = 0; j < N; j++)
                    {
                        summ += coofs[b].at<uchar>(i, j)*cos(CV_PI*v * (2 * i + 1) / (2 * N))*cos(CV_PI*v1 * (2 * j + 1) / (2 * N));

                    }
                }
                double znach = (sigma1(v)*sigma1(v1) / (sqrt(2 * N)))*summ;
                s[v][v1] = znach;



            }
        }
        sigma.push_back(s);
    }
    //коэффициенты
    x1 = 4;
    y1 = 5;
    x2 = 5;
    y2 = 4;

    //встраивание
    vector<double**> sigmaM;
    sigmaM = sigma;
//    double om1;
//    double om2;
//    double z1;
//    double z2;
    for (j = 0; j < length; j++)
    {

        for (i = 0; i < N; i++)
        {
            double ** sigmas = new double*[N];
            for (int m = 0; m < N; m++)
            {
                double* si = new double[N];
                sigmas[m] = si;

            }

            for (int i1 = 0; i1 < N; i1++)
            {
                for (int i2 = 0; i2 < N; i2++)
                {
                    sigmas[i1][i2] = sigma[N*j + i][i1][i2];

                }
            }
            om1 = fabs(sigmas[x1][y1]);
            om2 = fabs(sigmas[x2][y2]);
            if (sigmas[x1][y1] >= 0)
            {
                z1 = 1;
            }
            if (sigmas[x1][y1] < 0)
            {
                z1 = -1;
            }
            if (sigmas[x2][y2] >= 0)
            {
                z2 = 1;
            }
            if (sigmas[x2][y2] < 0)
            {
                z2 = -1;
            }

            if ((M[j][i] == 0) && (om1 - om2 <= P))
            {
                om1 = P / 2 + om2 + 1;
                om2 -= P / 2;
            }
            if ((M[j][i] == 1) && (om1 - om2 >= -P))
            {
                om2 = P / 2 + om1 + 1;
                om1 -= P / 2;
            }
            sigmas[x1][y1] = z1*om1;
            sigmas[x2][y2] = z2*om2;
            sigmaM[j * N + i] = sigmas;
        }


    }
    //обратное ДКП
    vector<double**>Cms;
    double summ, znach;
    for (int b = 0; b < Nc; b++)
    {
        double** s = new double*[N];
        for (int m = 0; m < N; m++)
        {
            double* si = new double[N];
            s[m] = si;

        }

        for (int x = 0; x < N; x++)
        {
            for (int y = 0; y < N; y++)
            {
                summ = 0;
                for (i = 0; i < N; i++)
                {
                    for (j = 0; j < N; j++)
                    {
                        summ += sigma1(i)*sigma1(j)*sigmaM[b][i][j] * cos(CV_PI*i * (2 * x + 1) / (2 * N))*cos(CV_PI*j * (2 * y + 1) / (2 * N));

                    }
                }
                znach = summ / (sqrt(2 * N));
                s[x][y] = znach;

            }
        }
        Cms.push_back(s);
    }

    //нахождение минимума и максимума для нормировки
    vector<double>max1;
    vector<double>min1;
    double maxv = Cms[0][0][0];
    double minv = Cms[0][0][0];
    for (i = 0; i < Nc; i++)
    {

        for (j = 0; j < N; j++)
        {

            for (k = 0; k < N; k++)
            {
                if (Cms[i][j][k]>maxv)
                {
                    maxv = Cms[i][j][k];
                }
                if (Cms[i][j][k] < minv)
                {
                    minv = Cms[i][j][k];

                }


            }
        }

    }


    //нормировка и присвоение значений

    int c1 = 0;
    int r1 = 0;
    double temp2;
    for (i = 0; i < Nc; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                temp2 = (Cms[i][j][k] + minv) * 255 / (minv + maxv);
                if ((c1 + j < heigh) && (r1 + k < widt))
                {
                    Matvector[0].at<uchar>(c1 + j, r1 + k) = (uchar)temp2;
                }

            }
        }
        r1 += N;
        if (r1 >= widt)
        {
            c1 += N;
            r1 = 0;
        }
    }

    vector<cv::Mat>Vec;
    Vec.push_back(Matvector[0]);
    Vec.push_back(Matvector[1]);
    Vec.push_back(Matvector[2]);
    cv::Mat FResult(heigh, widt, CV_8UC3);
    merge(Vec, FResult);
    t1 = clock() - t1;

    ui->duration->setText(QString::number(t1 / CLOCKS_PER_SEC) + " sec");
    //    cout << "Время встраивания ЦВЗ: " << (double)t1 / CLOCKS_PER_SEC << " секунд" << endl;
    //    namedWindow("Изображение с ЦВЗ", cv::WINDOW_AUTOSIZE);
    //    imshow("Изображение с ЦВЗ", FResult);
    //    cv::waitKey(0);
    //    cv::destroyWindow("Изображение с ЦВЗ");
    this->algResult = FResult;
    this->FResult   = FResult;
    imageProcessedPixels = cvMatToQPixmap(FResult);
    ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);

    int md = MD(start, Matvector[0]);
    double ad = AD(start, Matvector[0]);
    double nad = NAD(start, Matvector[0]);
    double mse = MSE(start, Matvector[0]);
    double nmse = NMSE(start, Matvector[0]);
    double snr = SNR(start, Matvector[0]);
    double psnr = PSNR(start, Matvector[0]);
    double If = IF(start, Matvector[0]);
    //    string merged = first + "Koch." + second;

    setQualityInfo(md, ad, nad, mse, nmse, snr, psnr, If);
}

void MainWindow::on_soheiliAlgorithm_clicked()
{
    setCurrentAlgorithm("soheili");
//    cv::Mat imag = QPixmapToCvMat(this->imagePixels);
    cv::Mat imag = QtOcv::image2Mat(imagePixels.toImage(), CV_8UC3, QtOcv::MCO_BGR);
    this->text = ui->signature->text().toStdString();

    Q = 10; //шаг квантования
//    cout << "Введите шаг квантования" << endl;
//    cin >> Q;
    int i, j, k;
    int channels = imag.channels();
    cv::Mat image;
    imag.convertTo(imag, CV_32F, 1.0, 0.0);
//    cv::Mat Matvector[3];

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
    vector<bitset<8>> B1; //битовое сообщение
    int length = text.length();
    uchar temp;
    for (i = 0; i < length; i++)
    {
        temp = (uchar)text[i];
        bitset<8>p((temp));
        B1.push_back(p);
    }
    int CVZsize = ceil(sqrt(length * 8));
    cv::Mat CVZ(CVZsize, CVZsize, CV_8U); //демонстрация ЦВЗ
    for (i = 0; i < CVZsize; i++)
    {
        for (j = 0; j < CVZsize; j++)
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
//    imwrite("CVZ.jpg", CVZ);
//    namedWindow(" ЦВЗ", WINDOW_AUTOSIZE);
//    imshow(" ЦВЗ", CVZ);
//    waitKey(0);
//    destroyWindow("ЦВЗ");



    //вейвлет-разложение контейнера
    clock_t t1 = clock();
    vector< cv::Mat > L1, L2;
    L1 = WaveletDec(image);
    L2 = WaveletDec(L1[0]);
    cv::Mat LL=L2[0].reshape(1, 1);

    //разбиение на подблоки
    wsize = text.length() * 8;
    int n = 2; //уровень разложения
    K = LL.cols / wsize/ (n*n);
    cv::Mat LL1 = LL;
    int m; //множитель
    //встраивание
    for (k = 0; k < K; k++)
    {
        for (i = 0; i < text.length(); i++)
        {

            for (j = 0; j < 8; j++)
            {
                m = LL.at <float>(k*wsize+i*8+j) / Q;
                if (B1[i][j] == 1)
                {
                    if ((LL.at <float>(k*wsize + i * 8 + j) > m*Q) && (LL.at <float>(k*wsize + i * 8 + j) <= (m + 0.5)*Q))
                    {
                        LL1.at <float>(k*wsize + i * 8 + j) = m*Q;
                    }

                    if ((LL.at <float>(k*wsize + i * 8 + j) > (m+0.5)*Q) && (LL.at <float>(k*wsize + i * 8 + j) <= (m +1)*Q))
                    {
                        LL1.at <float>(k*wsize + i * 8 + j) = (m+1)*Q;
                    }

                }
                if (B1[i][j] == 0)
                {
                    LL1.at <float>(k*wsize + i * 8 + j) = (m + 0.5)*Q;

                }

            }

        }
    }
    LL1 = LL1.reshape(1, L2[0].rows);

    //вейвлет-восстановление
    vector <cv::Mat> IL1, IL2;
    IL2.push_back(LL1);
    for (int i = 1; i < 4; i++)
    {
        IL2.push_back(L2[i]);
    }
    cv::Mat temp2 = WaveletRec(IL2, L1[0].rows,L1[0].cols);
    IL1.push_back(temp2);
    for (int i = 1; i < 4; i++)
    {
        IL1.push_back(L1[i]);
    }

    this->RW = WaveletRec(IL1, image.rows, image.cols);
    t1 = clock() - t1;
//    cout << "Время встраивания ЦВЗ: " << (double)t1 / CLOCKS_PER_SEC << " секунд" << endl;

    ui->duration->setText(QString::number(t1 / CLOCKS_PER_SEC) + " sec");

    cv::Mat RWI;
    this->RW.convertTo(RWI, CV_8U);
    cv::Mat FResult;
    string merged = first + "Soheili." + second;
    if (channels == 1) //чёрно-белое
    {
//        namedWindow("Wavelet Reconstruction", 1);
//        imshow("Wavelet Reconstruction", RWI);
//        waitKey(0);
//        this->imageProcessedPixels = cvMatToQPixmap(RWI);


        this->imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(RWI, QtOcv::MCO_BGR));
        ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
        FResult = this->RW;

//        imageProcessedPixels = cvMatToQPixmap(FResult);
//        this->imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult, QtOcv::MCO_BGR).rgbSwapped());
//        imwrite(merged, FResult);
    }
    if (channels == 3) //цветное
    {
        vector<cv::Mat>Vec;
        Vec.push_back(this->RW);
        Vec.push_back(Matvector[1]);
        Vec.push_back(Matvector[2]);
        merge(Vec, FResult);
        cv::Mat Fresult1;
        FResult.convertTo(Fresult1, CV_8UC3);

//        this->imageProcessedPixels = cvMatToQPixmap(Fresult1);

        this->imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(Fresult1, QtOcv::MCO_BGR));
        ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);

//        imageProcessedPixels = cvMatToQPixmap(FResult);
//        this->imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult, QtOcv::MCO_BGR, QImage::Format_RGB888));
//        namedWindow("Wavelet Reconstruction", 1);
//        imshow("Wavelet Reconstruction", Fresult1);
//        waitKey(0);
//        imwrite(merged, Fresult1);
    }

    this->FResult = FResult;
    this->algResult = FResult;
    //проверка качества
    int md = MD(charimage, RWI);
    double ad = AD(charimage, RWI);
    double nad = NAD(charimage, RWI);
    double mse = MSE(charimage, RWI);
    double nmse = NMSE(charimage, RWI);
    double snr = SNR(charimage, RWI);
    double psnr = PSNR(charimage, RWI);
    double If = IF(charimage, RWI);
    setQualityInfo(md, ad, nad, mse, nmse, snr, psnr, If);
}

void MainWindow::on_sanghaviAlgorithm_clicked()
{
    setCurrentAlgorithm("sanghavi");
    cv::Mat imag = QPixmapToCvMat(this->imagePixels);
    string text = ui->signature->text().toUtf8().toStdString();

    int channels = imag.channels();
    cv::Mat image;
    imag.convertTo(imag, CV_32F, 1.0, 0.0);
//    cv::Mat Matvector[3];

    cv::Mat charimage;
    int i, j, k;

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

    //Sanghavi 4 уровня
    vector <cv::Mat> L1, L2, L3, L4;
    //генерирование ЦВЗ
    vector<bitset<8>> B1;
    int length = text.length();
    uchar temp;
    for (i = 0; i < length; i++)
    {
        temp = (uchar)text[i];
        bitset<8>p((temp));
        B1.push_back(p);
    }

    int CVZsize = ceil(sqrt(length * 8));
    cv::Mat CVZ(CVZsize, CVZsize, CV_8U);
    for (i = 0; i < CVZsize; i++)
    {
        for (j = 0; j < CVZsize; j++)
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

//    imwrite("CVZ.jpg", CVZ);
//    cv::namedWindow(" ЦВЗ", cv::WINDOW_AUTOSIZE);
//    imshow(" ЦВЗ", CVZ);
//    cv::waitKey(0);
//    cv::destroyWindow("ЦВЗ");

    //вейвлет-разложение
    clock_t t1 = clock();
    L1 = WaveletDec(image);
    L2 = WaveletDec(L1[0]);
    L3 = WaveletDec(L2[0]);

    //встраивание ЦВЗ
    int k1, km;
    cv::Mat array = L3[2].reshape(1, 1);
    qDebug() << "FUCK1";

    if (text.size() * 40 >= array.cols)
    {
//		cout << "Изображение мало для встраивания" << endl;
        int fg;
//		cin >> fg;
    }

    float max, min,temp1;
    for (i = 0; i < length; i++)
    {
        for (j = 0; j < 8; j++)
        {
            k1 = (i * 8 + j) * 5;
            km = k1;
            max = array.at<float>(k1);
            min = array.at<float>(k1);
            if (B1[i][j] == 1)
            {
                for (k = k1 + 1; k < k1 + 5; k++)
                {
                    if (array.at<float>(k) > max)
                    {
                        max = array.at<float>(k);
                        km = k;

                    }

                }
            }
            if (B1[i][j] == 0)
            {
                for (k = k1 + 1; k < k1 + 5; k++)
                {
                    if (array.at<float>(k) < min)
                    {
                        min = array.at<float>(k);
                        km = k;

                    }

                }

            }
            temp1 = array.at<float>(k1);
            array.at<float>(k1) = array.at<float>(km);
            array.at<float>(km) = temp1;

        }
    }

    L3[2] = array.reshape(0, L3[1].rows);

    //вейвлет-восстановление
    vector <cv::Mat>LR1, LR2, LR3, LR4;
//    cv::Mat imr;
    LR3 = L3;
    cv::Mat LL2 = WaveletRec(LR3, L2[0].rows, L2[0].cols);
    LR2.push_back(LL2);
    for (i = 1; i <= 3; i++)
    {
    LR2.push_back(L2[i]);
    }

    cv::Mat LL1 = WaveletRec(LR2, L1[0].rows, L1[0].cols);
    LR1.push_back(LL1);
    for (i = 1; i <= 3; i++)
    {
    LR1.push_back(L1[i]);
    }
    imr = WaveletRec(LR1, rows, cols);
    t1 = clock() - t1;

//	cout << "Время встраивания ЦВЗ: " << (double)t1 / CLOCKS_PER_SEC << " секунд" << endl;
    ui->duration->setText(QString::number(t1 / CLOCKS_PER_SEC) + " sec");
    cv::Mat imrs;
    imr.convertTo(imrs, CV_8U);
    cv::Mat FResult;
    string merged = first + "Sanghavi." + second;

    if (channels == 1)
    {
        imageProcessedPixels = cvMatToQPixmap(imrs);
        ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
        this->width  = imrs.rows;
        this->height = imrs.cols;
//        this->imr = imr;
        this->FResult = FResult;
        this->algResult = FResult;
        qDebug() << "encode rows" << imr.rows;
//        imwrite(merged, FResult);
//        imageProcessedPixels = cvMatToQPixmap(FResult);
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

        imageProcessedPixels = cvMatToQPixmap(Fresult1);
        ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
        this->width  = Fresult1.rows;
        this->height = Fresult1.cols;
        this->FResult = FResult;
        this->algResult = FResult;
//        qDebug() << "encode rows" << imr.rows;

//        imwrite(merged, Fresult1);
//        imageProcessedPixels = cvMatToQPixmap(FResult);
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
//	cout << endl;
//	cout << "Показатели визуального искажения" << endl;
//	cout << "Максимальная разность значений пикселов: " << md << endl;
//	cout << "Средняя абсолютная разность значений пикселов: " << ad << endl;
//	cout << "Нормированная средняя абсолютная разность: " << nad << endl;
//	cout << "Отношение сигнал-шум: " << snr << endl;
//	cout << "Максимальное отношение сигнал-шум: " << psnr << endl;
//	cout << "Качество изображения: " << If * 100 << "%" << endl;
    setQualityInfo(md, ad, nad, mse, nmse, snr, psnr, If);
}



// ====ATTACKS====
cv::Mat Turn(cv::Mat src)//поворот на 60 градусов
{
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, src.type());
    cv::Mat rot_mat(2, 3, CV_32FC1);
    cv::Point center = cv::Point(src.rows / 2, src.cols / 2);
    double angle = -90.0;
    double scale = 1;
    rot_mat = getRotationMatrix2D(center, angle, scale);
    warpAffine(src, dst, rot_mat, dst.size());//преобразование поворотом
    namedWindow("Turned", cv::WINDOW_AUTOSIZE);
    cv::Mat dst1;
    if (dst.channels() == 1)
    {
        dst.convertTo(dst1, CV_8U);

    }
    if (dst.channels() == 3)
    {
        dst.convertTo(dst1, CV_8UC3);

    }
//    imshow("Turned", dst1);
//    waitKey(0);
//    destroyWindow("Turned");
    return dst;
}

cv::Mat Resize(cv::Mat src, int heigh, int widt)//изменение размера
{
    cv::Mat dst;
    resize(src, dst, cv::Size(), 0.5, 0.5, cv::INTER_AREA);//уменьшаем размер в 2 раза
    heigh = heigh / 2;
    widt = widt / 2;
    cv::Mat dst1;
    if (dst.channels() == 1)
    {
        dst.convertTo(dst1, CV_8U);

    }
    if (dst.channels() == 3)
    {
        dst.convertTo(dst1, CV_8UC3);

    }
//    namedWindow("Resized", WINDOW_AUTOSIZE);
//    imshow("Resized", dst1);
//    waitKey(0);
//    destroyWindow("Resized");

    qDebug() << "RESIZE" << heigh << widt;

    return dst;
}

cv::Mat Gaussian(cv::Mat src)//сглаживание при помощи функции Гаусса
{
    cv::Mat dst;
    GaussianBlur(src, dst, cv::Size(3, 3), 0, 0);
    cv::Mat dst1;
    if (dst.channels() == 1)
    {
        dst.convertTo(dst1, CV_8U);

    }
    if (dst.channels() == 3)
    {
        dst.convertTo(dst1, CV_8UC3);

    }
//    namedWindow("Gauss", WINDOW_AUTOSIZE);
//    imshow("Gauss", dst1);
//    waitKey(0);
//    destroyWindow("Gauss");
    return dst;
}

cv::Mat Chetkost(cv::Mat src)//повышение чёткости
{
    cv::Mat dst;
    float kernel[9];
    kernel[0] = -0.1;
    kernel[1] = -0.1;
    kernel[2] = -0.1;
    kernel[3] = -0.1;
    kernel[4] = 2;
    kernel[5] = -0.1;
    kernel[6] = -0.1;
    kernel[7] = -0.1;
    kernel[8] = -0.1;
    cv::Mat kernel_matrix(3, 3, CV_32FC1, kernel);
    filter2D(src, dst, -1, kernel_matrix, cv::Point(-1, 1), 0, cv::BORDER_DEFAULT);
    cv::Mat dst1;
    if (dst.channels() == 1)
    {
        dst.convertTo(dst1, CV_8U);

    }
    if (dst.channels() == 3)
    {
        dst.convertTo(dst1, CV_8UC3);

    }
//    namedWindow("Chetkost", WINDOW_AUTOSIZE);
//    imshow("Chetkost", dst1);
//    waitKey(0);
//    destroyWindow("Chetkost");
    return dst;
}

cv::Mat brightness(cv::Mat src)//увеличение яркости
{
    cv::Mat dst;
    float kernel[9];
    kernel[0] = -0.1;
    kernel[1] = 0.2;
    kernel[2] = -0.1;

    kernel[3] = 0.2;
    kernel[4] = 3;
    kernel[5] = 0.2;

    kernel[6] = -0.1;
    kernel[7] = 0.2;
    kernel[8] = -0.1;

    cv::Mat kernel_matrix(3, 3, CV_32FC1, kernel);
    filter2D(src, dst, -1, kernel_matrix, cv::Point(-1, 1), 0, cv::BORDER_DEFAULT);
    cv::Mat dst1;

    if (dst.channels() == 1)
    {
        dst.convertTo(dst1, CV_8U);

    }
    if (dst.channels() == 3)
    {
        dst.convertTo(dst1, CV_8UC3);

    }
//    namedWindow("Bright", WINDOW_AUTOSIZE);
//    imshow("Bright", dst1);
//    waitKey(0);
//    destroyWindow("Bright");
    return dst;
}

cv::Mat Dark(cv::Mat src)//уменьшение яркости
{
    cv::Mat dst;
    float kernel[9];
    kernel[0] = -0.1;
    kernel[1] = 0.1;
    kernel[2] = -0.1;

    kernel[3] = 0.1;
    kernel[4] = 0.5;
    kernel[5] = 0.1;

    kernel[6] = -0.1;
    kernel[7] = 0.1;
    kernel[8] = -0.1;
    cv::Mat kernel_matrix(3, 3, CV_32FC1, kernel);
    filter2D(src, dst, -1, kernel_matrix, cv::Point(-1, 1), 0, cv::BORDER_DEFAULT);
    cv::Mat dst1;
    if (dst.channels() == 1)
    {
        dst.convertTo(dst1, CV_8U);

    }
    if (dst.channels() == 3)
    {
        dst.convertTo(dst1, CV_8UC3);

    }
//    namedWindow("Dark", WINDOW_AUTOSIZE);
//    imshow("Dark", dst1);
//    waitKey(0);
//    destroyWindow("Dark");
    return dst;
}

cv::Mat Erode(cv::Mat src)
{
    cv::Mat dst;
    int radius = 3;
    cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * radius + 1, 2 * radius + 1), cv::Point(radius, radius));
    erode(src, dst, element);
    cv::Mat dst1;
    if (dst.channels() == 1)
    {
        dst.convertTo(dst1, CV_8U);

    }
    if (dst.channels() == 3)
    {
        dst.convertTo(dst1, CV_8UC3);

    }
//    namedWindow("Erode", cv::WINDOW_AUTOSIZE);
//    imshow("Erode", dst1);
//    cv::waitKey(0);
//    cv::destroyWindow("Erode");
    return dst;
}

cv::Mat CutRight(cv::Mat src, int heigh, int widt)
{
    src.convertTo(src, CV_32FC3, 1.0, 0.0);
    int i, j;
    int news = widt*0.8;
    cv::Mat dst;



    if (src.channels() == 3)
    {
        cv::Mat dst1(heigh, news, CV_32FC3);

        for (i = 0; i < heigh; i++)
        {
            for (j = 0; j < news; j++)
            {
                dst1.at<float>(i, j) = src.at<float>(i, j);
            }
        }

        dst = dst1;
    }

    if (src.channels() == 1)
    {
        cv::Mat dst2(heigh, news, CV_32FC1);
        for (i = 0; i < heigh; i++)
        {
            for (j = 0; j < news; j++)
            {
                dst2.at<float>(i, j) = src.at<float>(i, j);
            }
        }
        dst = dst2;
    }

    cv::Mat dst3;
    if (dst.channels() == 1)
    {
        dst.convertTo(dst3, CV_8U);

    }
    if (dst.channels() == 3)
    {
        dst.convertTo(dst3, CV_8UC3);

    }

    namedWindow("Cut", cv::WINDOW_AUTOSIZE);
    imshow("Cut", dst3);
    cv::waitKey(0);
//    cv::destroyWindow("Cut");

    qDebug() << "CUTRIGHT" << heigh << widt;
    return dst3;
}

cv::Mat CutDown(cv::Mat src, int heigh, int widt)
{

    src.convertTo(src, CV_32FC3, 1.0, 0.0);
    int i, j;
    int news = heigh*0.8;
    cv::Mat dst;


    if (src.channels() == 3)
    {
        qDebug() << src.channels();

        cv::Mat dst1(news, widt, CV_32FC3);

        for (i = 0; i < news; i++)
        {
            for (j = 0; j < widt * 3; j++)
            {
                dst1.at<float>(i, j) = src.at<float>(i, j);
            }
        }

        dst = dst1;
    }

    qDebug() << "this 2";

    if (src.channels() == 1)
    {
        cv::Mat dst2(news, widt, CV_32FC1);

        for (i = 0; i < news; i++)
        {
            for (j = 0; j < widt * 3; j++)
            {
                dst2.at<float>(i, j) = src.at<float>(i, j);
            }
        }
        dst = dst2;
    }

    cv::Mat dst3;

    qDebug() << "this 3";

    if (dst.channels() == 1)
    {
        dst.convertTo(dst3, CV_8U);

    }
    if (dst.channels() == 3)
    {
        dst.convertTo(dst3, CV_8UC3);

    }

    //    namedWindow("Cut", cv::WINDOW_AUTOSIZE);
    //    imshow("Cut", dst3);
    //    cv::waitKey(0);
    //    cv::destroyWindow("Cut");

        qDebug() << "CUTDOWN" << heigh << widt;
    return dst3;
}

cv::Mat JPEGComp(cv::Mat src)
{
    vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    int x = 40;
//    cout << "Введите коэффициент сжатия в процентах" << endl;
//    cin >> x;
    compression_params.push_back(x);
    cv::Mat src1;
    cv::Mat dst;
    imwrite("Comp.jpg", src, compression_params);
    dst = imread("Comp.jpg", cv::IMREAD_COLOR);
    cv::Mat dst1;
    if (dst.channels() == 1)
    {
        dst.convertTo(dst1, CV_8U);
    }

    if (src.channels() == 3)
    {
        dst.convertTo(dst1, CV_8UC3);
    }
//    cv::imshow("Comp.jpg", dst1);
//    cv::waitKey(0);
//    cv::destroyWindow("Comp.jpg");
    return dst;

}


void MainWindow::on_jpegCompression_clicked()
{
    //    cv::Mat FResult = JPEGComp(QPixmapToCvMat(imageProcessedPixels));
    //    cv::Mat FResult = JPEGComp(QtOcv::image2Mat(imageProcessedPixels.toImage(), CV_8UC3, QtOcv::MCO_RGB));

    this->FResult = JPEGComp(this->algResult);

    qDebug() << "Compressed";
    //    imageProcessedPixels = cvMatToQPixmap(FResult);
    imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult, QtOcv::MCO_BGR));

    ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
    this->width  = FResult.rows;
    this->height = FResult.cols;
}

void MainWindow::on_Dark_clicked()
{
    this->FResult = Dark(this->algResult);
    cv::Mat FResult = Dark(QtOcv::image2Mat(imageProcessedPixels.toImage(), CV_8UC3));
    qDebug() << "dark";
    imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult));
    ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
    this->width  = FResult.rows;
    this->height = FResult.cols;
}

void MainWindow::on_resize_clicked()
{
    this->FResult = Resize(this->algResult, this->height, this->width);
    cv::Mat FResult = Resize(QtOcv::image2Mat(imageProcessedPixels.toImage(), CV_8UC3), this->height, this->width);
    imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult));
    ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
    this->width  = FResult.rows;
    this->height = FResult.cols;
}

void MainWindow::on_turn_clicked()
{
    this->FResult = Turn(this->algResult);
    cv::Mat FResult = Turn(QtOcv::image2Mat(imageProcessedPixels.toImage(), CV_8UC3));
    imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult));
    ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
    this->width  = FResult.rows;
    this->height = FResult.cols;
}

void MainWindow::on_gaussian_clicked()
{
    this->FResult = Gaussian(this->algResult);
    cv::Mat FResult = Gaussian(QtOcv::image2Mat(imageProcessedPixels.toImage(), CV_8UC3));
    imageProcessedPixels = cvMatToQPixmap(FResult);
    ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
    this->width  = FResult.rows;
    this->height = FResult.cols;
}

void MainWindow::on_checkost_clicked()
{
    this->FResult = Chetkost(this->algResult);
    cv::Mat FResult = Chetkost(QtOcv::image2Mat(imageProcessedPixels.toImage(), CV_8UC3));
    qDebug() << "KEK";
    imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult));
    ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
    this->width  = FResult.rows;
    this->height = FResult.cols;
}

void MainWindow::on_brightness_clicked()
{
    this->FResult = brightness(this->algResult);
    cv::Mat FResult = brightness(QtOcv::image2Mat(imageProcessedPixels.toImage(), CV_8UC3));
    imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult));
    ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
    this->width  = FResult.rows;
    this->height = FResult.cols;
}

void MainWindow::on_erode_clicked()
{
    this->FResult = Erode(this->algResult);
    cv::Mat FResult = Erode(QtOcv::image2Mat(imageProcessedPixels.toImage(), CV_8UC3));
    imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult));
    ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
    this->width  = FResult.rows;
    this->height = FResult.cols;
}

void MainWindow::on_cutRight_clicked()
{
    this->FResult = CutRight(this->algResult, this->height, this->width);
    cv::Mat FResult = CutRight(QtOcv::image2Mat(imageProcessedPixels.toImage(), CV_8UC3), this->height, this->width);
    imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult));
    ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
    this->width  = FResult.rows;
    this->height = FResult.cols;
}

void MainWindow::on_cutDown_clicked()
{
    this->FResult = CutDown(this->algResult, this->height, this->width);
    cv::Mat FResult = CutDown(QtOcv::image2Mat(imageProcessedPixels.toImage(), CV_8UC3), this->height, this->width);
    imageProcessedPixels = QPixmap::fromImage(QtOcv::mat2Image(FResult));
    ui->ImageProcessedWrap->setPixmap(imageProcessedPixels);
    this->width  = FResult.rows;
    this->height = FResult.cols;
}


void MainWindow::on_decode_clicked()
{
    decode();
}

// ====ADDITIONAL FUNCTIONS=======


vector<cv::Mat> MainWindow::WaveletDec(cv::Mat image)
{
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
            c = (a + b) * 0.707;
            im11.at<float>(rcnt, ccnt) = c;
            d = (a - b) * 0.707;                     //Filtering at Stage I
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



void MainWindow::on_pushButton_clicked()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"), QDir::homePath(), tr("JPG files (*.jpg)"));

    try {
        imwrite(fileName.toStdString(), this->algResult);
        QMessageBox::information(this, "Success", "File " + fileName + " was created");
    } catch(const cv::Exception& ex){
        QMessageBox::information(this, "Warning", "Error with saving");
        qDebug() << "Exception converting image to PNG format: %s\n" << ex.what();
    }

}
