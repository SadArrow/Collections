/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QLabel *hi;
    QWidget *Cat;
    QPushButton *Button1;
    QPushButton *Button3;
    QPushButton *Button4;
    QPushButton *Button8;
    QPushButton *Button6;
    QPushButton *Button5;
    QPushButton *Button7;
    QPushButton *Button2;
    QPushButton *Exit;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(433, 369);
        MainWindow->setStyleSheet(QString::fromUtf8("#calendarwin{border-image:url(:/img/image/calendar_background.png)}"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        hi = new QLabel(centralwidget);
        hi->setObjectName("hi");
        hi->setGeometry(QRect(190, 40, 91, 81));
        Cat = new QWidget(centralwidget);
        Cat->setObjectName("Cat");
        Cat->setGeometry(QRect(80, 80, 161, 151));
        Cat->setStyleSheet(QString::fromUtf8("border-image: url(:/img/image/cat1.png);\n"
""));
        Button1 = new QPushButton(centralwidget);
        Button1->setObjectName("Button1");
        Button1->setGeometry(QRect(100, 30, 41, 41));
        Button1->setStyleSheet(QString::fromUtf8("border-image: url(:/img/image/button1.png);"));
        Button3 = new QPushButton(centralwidget);
        Button3->setObjectName("Button3");
        Button3->setGeometry(QRect(30, 90, 41, 41));
        Button3->setStyleSheet(QString::fromUtf8("border-image: url(:/img/image/button3.png);"));
        Button4 = new QPushButton(centralwidget);
        Button4->setObjectName("Button4");
        Button4->setGeometry(QRect(30, 170, 41, 41));
        Button4->setStyleSheet(QString::fromUtf8("border-image: url(:/img/image/button4.png);"));
        Button8 = new QPushButton(centralwidget);
        Button8->setObjectName("Button8");
        Button8->setGeometry(QRect(240, 170, 41, 41));
        Button8->setStyleSheet(QString::fromUtf8("border-image:url(:/img/image/button8.png)"));
        Button6 = new QPushButton(centralwidget);
        Button6->setObjectName("Button6");
        Button6->setGeometry(QRect(170, 230, 41, 41));
        Button6->setStyleSheet(QString::fromUtf8("border-image: url(:/img/image/button6.png);"));
        Button5 = new QPushButton(centralwidget);
        Button5->setObjectName("Button5");
        Button5->setGeometry(QRect(100, 230, 41, 41));
        Button5->setStyleSheet(QString::fromUtf8("border-image: url(:/img/image/button5.png);"));
        Button7 = new QPushButton(centralwidget);
        Button7->setObjectName("Button7");
        Button7->setGeometry(QRect(240, 90, 41, 41));
        Button7->setStyleSheet(QString::fromUtf8("border-image: url(:/img/image/button7.png);"));
        Button2 = new QPushButton(centralwidget);
        Button2->setObjectName("Button2");
        Button2->setGeometry(QRect(170, 30, 41, 41));
        Button2->setStyleSheet(QString::fromUtf8("border-image: url(:/img/image/button2.png);"));
        Exit = new QPushButton(centralwidget);
        Exit->setObjectName("Exit");
        Exit->setGeometry(QRect(0, 40, 41, 41));
        Exit->setStyleSheet(QString::fromUtf8("\n"
"border-image: url(:/img/feeding_Img/exit.png);"));
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 433, 17));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        hi->setText(QString());
        Button1->setText(QString());
        Button3->setText(QString());
        Button4->setText(QString());
        Button8->setText(QString());
        Button6->setText(QString());
        Button5->setText(QString());
        Button7->setText(QString());
        Button2->setText(QString());
        Exit->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
