/********************************************************************************
** Form generated from reading UI file 'feed.ui'
**
** Created by: Qt User Interface Compiler version 6.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FEED_H
#define UI_FEED_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QTableWidget>

QT_BEGIN_NAMESPACE

class Ui_feed
{
public:
    QTableWidget *foodtable;
    QLabel *eatapple_catfood;
    QLabel *eatchicken;
    QLabel *toeat;
    QLabel *eatfish;

    void setupUi(QDialog *feed)
    {
        if (feed->objectName().isEmpty())
            feed->setObjectName("feed");
        feed->resize(400, 300);
        foodtable = new QTableWidget(feed);
        if (foodtable->columnCount() < 2)
            foodtable->setColumnCount(2);
        if (foodtable->rowCount() < 2)
            foodtable->setRowCount(2);
        foodtable->setObjectName("foodtable");
        foodtable->setGeometry(QRect(20, 50, 100, 101));
        foodtable->setLayoutDirection(Qt::LeftToRight);
        foodtable->setStyleSheet(QString::fromUtf8("border-image: url(:/img/feeding_Img/2.png);"));
        foodtable->setFrameShape(QFrame::NoFrame);
        foodtable->setFrameShadow(QFrame::Plain);
        foodtable->setRowCount(2);
        foodtable->setColumnCount(2);
        foodtable->horizontalHeader()->setVisible(false);
        foodtable->horizontalHeader()->setMinimumSectionSize(50);
        foodtable->horizontalHeader()->setDefaultSectionSize(50);
        foodtable->horizontalHeader()->setProperty("showSortIndicator", QVariant(false));
        foodtable->verticalHeader()->setVisible(false);
        foodtable->verticalHeader()->setMinimumSectionSize(50);
        foodtable->verticalHeader()->setDefaultSectionSize(50);
        eatapple_catfood = new QLabel(feed);
        eatapple_catfood->setObjectName("eatapple_catfood");
        eatapple_catfood->setGeometry(QRect(210, 20, 161, 171));
        eatapple_catfood->setScaledContents(false);
        eatchicken = new QLabel(feed);
        eatchicken->setObjectName("eatchicken");
        eatchicken->setGeometry(QRect(170, 20, 181, 171));
        eatchicken->setScaledContents(false);
        toeat = new QLabel(feed);
        toeat->setObjectName("toeat");
        toeat->setGeometry(QRect(210, 20, 161, 171));
        toeat->setScaledContents(false);
        eatfish = new QLabel(feed);
        eatfish->setObjectName("eatfish");
        eatfish->setGeometry(QRect(150, 20, 221, 181));
        eatfish->setAlignment(Qt::AlignCenter);

        retranslateUi(feed);

        QMetaObject::connectSlotsByName(feed);
    } // setupUi

    void retranslateUi(QDialog *feed)
    {
        feed->setWindowTitle(QCoreApplication::translate("feed", "Dialog", nullptr));
        eatapple_catfood->setText(QString());
        eatchicken->setText(QString());
        toeat->setText(QString());
        eatfish->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class feed: public Ui_feed {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FEED_H
