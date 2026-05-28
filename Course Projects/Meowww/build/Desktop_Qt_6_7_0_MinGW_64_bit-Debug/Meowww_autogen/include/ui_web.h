/********************************************************************************
** Form generated from reading UI file 'web.ui'
**
** Created by: Qt User Interface Compiler version 6.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WEB_H
#define UI_WEB_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_web
{
public:
    QPushButton *coursebtn;
    QPushButton *netdiskbtn;
    QPushButton *portalbtn;
    QPushButton *treeholebtn;

    void setupUi(QDialog *web)
    {
        if (web->objectName().isEmpty())
            web->setObjectName("web");
        web->resize(200, 60);
        coursebtn = new QPushButton(web);
        coursebtn->setObjectName("coursebtn");
        coursebtn->setGeometry(QRect(20, 10, 30, 30));
        coursebtn->setStyleSheet(QString::fromUtf8("QPushButton#coursebtn{\n"
"	\n"
"	border-image: url(:/img/web_Img/courses.png);\n"
"}"));
        netdiskbtn = new QPushButton(web);
        netdiskbtn->setObjectName("netdiskbtn");
        netdiskbtn->setGeometry(QRect(60, 10, 30, 30));
        netdiskbtn->setStyleSheet(QString::fromUtf8("QPushButton#netdiskbtn{\n"
"border-image: url(:/img/web_Img/netdisk.png);\n"
"}"));
        portalbtn = new QPushButton(web);
        portalbtn->setObjectName("portalbtn");
        portalbtn->setGeometry(QRect(100, 10, 30, 30));
        portalbtn->setStyleSheet(QString::fromUtf8("QPushButton#portalbtn{\n"
"border-image: url(:/img/web_Img/PKU_logo.png);\n"
"}"));
        treeholebtn = new QPushButton(web);
        treeholebtn->setObjectName("treeholebtn");
        treeholebtn->setGeometry(QRect(140, 10, 30, 30));
        treeholebtn->setStyleSheet(QString::fromUtf8("QPushButton#treeholebtn{\n"
"border-image: url(:/img/web_Img/treehole.png);\n"
"}"));

        retranslateUi(web);

        QMetaObject::connectSlotsByName(web);
    } // setupUi

    void retranslateUi(QDialog *web)
    {
        web->setWindowTitle(QCoreApplication::translate("web", "Dialog", nullptr));
#if QT_CONFIG(whatsthis)
        coursebtn->setWhatsThis(QString());
#endif // QT_CONFIG(whatsthis)
        coursebtn->setText(QString());
        netdiskbtn->setText(QString());
        portalbtn->setText(QString());
        treeholebtn->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class web: public Ui_web {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WEB_H
