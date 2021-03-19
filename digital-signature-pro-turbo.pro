QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui



INCLUDEPATH += D:\opencv\build\include

#@LIBS += LD:\opencv\opencv-build\install\x64\mingw\lib -lopencv_core451 -lopencv_highgui451 -lopencv_imgcodec451 -lopencv_imgproc451 -lopencv_features2d451 -lopencv_calib3d451

#LIBS += D:\opencv\opencv-build\bin\libopencv_core451.dll
#LIBS += D:\opencv\opencv-build\bin\libopencv_highgui451.dll
#LIBS += D:\opencv\opencv-build\bin\libopencv_imgcodecs451.dll
#LIBS += D:\opencv\opencv-build\bin\libopencv_imgproc451.dll
#LIBS += D:\opencv\opencv-build\bin\libopencv_features2d451.dll
#LIBS += D:\opencv\opencv-build\bin\libopencv_calib3d451.dll

LIBS += -LD:\opencv\opencv-build\install\x64\mingw\lib \
        -lopencv_core451        \
        -lopencv_highgui451     \
        -lopencv_imgcodecs451   \
        -lopencv_imgproc451     \
        -lopencv_features2d451  \
        -lopencv_calib3d451

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


=======
OPENCV_SDK_DIR=D:/opencv-build/install
INCLUDEPATH += $$(OPENCV_SDK_DIR)/include

LIBS += -L$$(OPENCV_SDK_DIR)/x86/mingw/lib \
        -lopencv_core451        \
        -lopencv_highgui451     \
        -lopencv_imgcodecs451   \
        -lopencv_imgproc451     \
        -lopencv_features2d451  \
        -lopencv_calib3d451
