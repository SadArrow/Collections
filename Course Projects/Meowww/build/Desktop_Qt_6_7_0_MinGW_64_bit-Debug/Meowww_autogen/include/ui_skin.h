/********************************************************************************
** Form generated from reading UI file 'skin.ui'
**
** Created by: Qt User Interface Compiler version 6.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SKIN_H
#define UI_SKIN_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_skin
{
public:
    QWidget *centralwidget;
    QPushButton *choose_skin_1;
    QPushButton *choose_skin_2;
    QPushButton *choose_skin_3;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *skin)
    {
        if (skin->objectName().isEmpty())
            skin->setObjectName("skin");
        skin->resize(169, 87);
        centralwidget = new QWidget(skin);
        centralwidget->setObjectName("centralwidget");
        choose_skin_1 = new QPushButton(centralwidget);
        choose_skin_1->setObjectName("choose_skin_1");
        choose_skin_1->setGeometry(QRect(10, 0, 41, 41));
        choose_skin_1->setStyleSheet(QString::fromUtf8("border-image: url(:/img/image/choose_cat1.png);"));
        choose_skin_2 = new QPushButton(centralwidget);
        choose_skin_2->setObjectName("choose_skin_2");
        choose_skin_2->setGeometry(QRect(60, 0, 41, 41));
        choose_skin_2->setStyleSheet(QString::fromUtf8("border-image: url(:/img/image/choose_cat2.png);"));
        choose_skin_3 = new QPushButton(centralwidget);
        choose_skin_3->setObjectName("choose_skin_3");
        choose_skin_3->setGeometry(QRect(110, 0, 41, 41));
        choose_skin_3->setStyleSheet(QString::fromUtf8("border-image: url(:/img/image/choose_cat3.png);"));
        skin->setCentralWidget(centralwidget);
        menubar = new QMenuBar(skin);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 169, 17));
        skin->setMenuBar(menubar);
        statusbar = new QStatusBar(skin);
        statusbar->setObjectName("statusbar");
        skin->setStatusBar(statusbar);

        retranslateUi(skin);

        QMetaObject::connectSlotsByName(skin);
    } // setupUi

    void retranslateUi(QMainWindow *skin)
    {
        skin->setWindowTitle(QCoreApplication::translate("skin", "skin", nullptr));
        choose_skin_1->setText(QString());
        choose_skin_2->setText(QString());
        choose_skin_3->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class skin: public Ui_skin {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SKIN_H
