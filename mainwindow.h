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
using namespace cv;
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

    vector<Mat> WaveletDec();
    Mat WaveletRec(vector<Mat> Decomp,int rows,int cols);
    vector<Mat> WaveletDec8();
    Mat WaveletRec8(vector<Mat> Decomp, int rows, int cols);
    double sigma1(int x);
    int MD(Mat cont, Mat stego);
    double AD(Mat cont, Mat stego);
    double NAD(Mat cont, Mat stego);
    double MSE(Mat cont, Mat stego);
    double NMSE(Mat cont, Mat stego);
    double SNR(Mat cont, Mat stego);
    double PSNR(Mat cont, Mat stego);
    double IF(Mat cont, Mat stego);

    QString imagePath;
    QPixmap imagePixels;

    cv::Mat mat;
};
#endif // MAINWINDOW_H
