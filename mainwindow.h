#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QImage>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <windows.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
//using namespace cv;
using namespace std;
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_loadImage_clicked();

    void on_authorAlgorithm_clicked();

    void on_sanghaviAlgorithm_clicked();

    void on_kochAlgorithm_clicked();

    void on_jpegCompression_clicked();

    void on_pushButton_10_clicked();

    void on_Dark_clicked();

    void on_resize_clicked();

    void on_turn_clicked();

    void on_gaussian_clicked();

    void on_checkost_clicked();

    void on_brightness_clicked();

    void on_erode_clicked();

    void on_cutRight_clicked();

    void on_pushButton_9_clicked();

    void on_decode_clicked();

    void on_cutDown_clicked();

private:
    Ui::MainWindow *ui;

    vector<cv::Mat> WaveletDec(cv::Mat image);
    cv::Mat WaveletRec(vector<cv::Mat> Decomp,int rows,int cols);
    vector<cv::Mat> WaveletDec8();
    cv::Mat WaveletRec8(vector<cv::Mat> Decomp, int rows, int cols);
    double sigma1(int x);
    int MD(cv::Mat cont, cv::Mat stego);
    double AD(cv::Mat cont, cv::Mat stego);
    double NAD(cv::Mat cont, cv::Mat stego);
    double MSE(cv::Mat cont, cv::Mat stego);
    double NMSE(cv::Mat cont, cv::Mat stego);
    double SNR(cv::Mat cont, cv::Mat stego);
    double PSNR(cv::Mat cont, cv::Mat stego);
    double IF(cv::Mat cont, cv::Mat stego);

    void setQualityInfo(int md, double ad, double nad, double mse, double nmse, double snr, double pnsr, double IF);
    void setCurrentAlgorithm(QString name);
    void decode();

    void decodeKoch();
    void decodeAuthor();
    void decodeSanghavi();
    void decodeSoheili();

//    cv::Mat
    QString imagePath;
    QString currentAlgorithm;
    QPixmap imagePixels;
    QPixmap imageProcessedPixels;
    string  first;
    string  second;
    int     width;
    int     height;

    cv::Mat Matvector[3];

    int x1;
    int y1;
    int x2;
    int y2;

    double om1;
    double om2;
    double z1;
    double z2;

    LARGE_INTEGER authorFrequency;
    LARGE_INTEGER authorT1;
    LARGE_INTEGER authorT2;
    int           authorN;
    int           author;

    cv::Mat mat;
};
#endif // MAINWINDOW_H
