#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QImage>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

private:
    Ui::MainWindow *ui;

    vector<cv::Mat> WaveletDec();
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

    QString imagePath;
    QPixmap imagePixels;
    string  first;
    string  second;

    cv::Mat mat;
};
#endif // MAINWINDOW_H
