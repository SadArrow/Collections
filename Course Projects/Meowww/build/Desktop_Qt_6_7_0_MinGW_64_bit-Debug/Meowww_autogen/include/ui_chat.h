/********************************************************************************
** Form generated from reading UI file 'chat.ui'
**
** Created by: Qt User Interface Compiler version 6.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CHAT_H
#define UI_CHAT_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_chat
{
public:
    QWidget *chatwidget;
    QPushButton *words4;
    QPushButton *words1;
    QPushButton *words3;
    QPushButton *words2;
    QPushButton *refreshbtn;
    QLabel *responselabel;
    QLabel *responsetext;
    QLabel *Cat;

    void setupUi(QDialog *chat)
    {
        if (chat->objectName().isEmpty())
            chat->setObjectName("chat");
        chat->resize(605, 466);
        chatwidget = new QWidget(chat);
        chatwidget->setObjectName("chatwidget");
        chatwidget->setGeometry(QRect(270, 160, 300, 220));
        chatwidget->setAutoFillBackground(false);
        chatwidget->setStyleSheet(QString::fromUtf8("QWidget#chatwidget{\n"
"	border-image: url(:/img/chat_Img/chatframe2.png);\n"
"}"));
        words4 = new QPushButton(chatwidget);
        words4->setObjectName("words4");
        words4->setGeometry(QRect(160, 50, 115, 41));
        QFont font;
        font.setFamilies({QString::fromUtf8("Eras Demi ITC")});
        font.setPointSize(11);
        words4->setFont(font);
        words4->setStyleSheet(QString::fromUtf8("border-image: url(:/img/chat_Img/chatframe3-1.png.png);"));
        words1 = new QPushButton(chatwidget);
        words1->setObjectName("words1");
        words1->setGeometry(QRect(30, 50, 115, 41));
        QPalette palette;
        words1->setPalette(palette);
        words1->setFont(font);
        words1->setStyleSheet(QString::fromUtf8("border-image: url(:/img/chat_Img/chatframe3-1.png.png);"));
        words1->setCheckable(false);
        words1->setAutoDefault(false);
        words1->setFlat(false);
        words3 = new QPushButton(chatwidget);
        words3->setObjectName("words3");
        words3->setGeometry(QRect(30, 110, 115, 41));
        words3->setFont(font);
        words3->setStyleSheet(QString::fromUtf8("border-image: url(:/img/chat_Img/chatframe3-1.png.png);"));
        words2 = new QPushButton(chatwidget);
        words2->setObjectName("words2");
        words2->setGeometry(QRect(160, 110, 115, 41));
        words2->setFont(font);
        words2->setStyleSheet(QString::fromUtf8("border-image: url(:/img/chat_Img/chatframe3-1.png.png);"));
        refreshbtn = new QPushButton(chatwidget);
        refreshbtn->setObjectName("refreshbtn");
        refreshbtn->setGeometry(QRect(240, 170, 31, 31));
        refreshbtn->setStyleSheet(QString::fromUtf8("border-image: url(:/img/chat_Img/refresh.png);"));
        responselabel = new QLabel(chat);
        responselabel->setObjectName("responselabel");
        responselabel->setGeometry(QRect(270, 0, 161, 71));
        responselabel->setStyleSheet(QString::fromUtf8("border-image: url(:/img/chat_Img/chatframe3.png);"));
        responselabel->setLineWidth(2);
        responselabel->setScaledContents(true);
        responselabel->setAlignment(Qt::AlignCenter);
        responselabel->setWordWrap(false);
        responsetext = new QLabel(chat);
        responsetext->setObjectName("responsetext");
        responsetext->setGeometry(QRect(300, 0, 131, 51));
        responsetext->setFont(font);
        responsetext->setStyleSheet(QString::fromUtf8(""));
        responsetext->setTextFormat(Qt::PlainText);
        Cat = new QLabel(chat);
        Cat->setObjectName("Cat");
        Cat->setGeometry(QRect(330, 0, 140, 160));
        Cat->setMaximumSize(QSize(16777215, 16777215));
        Cat->setPixmap(QPixmap(QString::fromUtf8(":/images/feeding_Img/miao.png")));
        Cat->setScaledContents(true);
        Cat->setAlignment(Qt::AlignCenter);
        Cat->raise();
        chatwidget->raise();
        responselabel->raise();
        responsetext->raise();
#if QT_CONFIG(shortcut)
#endif // QT_CONFIG(shortcut)

        retranslateUi(chat);

        words1->setDefault(false);


        QMetaObject::connectSlotsByName(chat);
    } // setupUi

    void retranslateUi(QDialog *chat)
    {
        chat->setWindowTitle(QCoreApplication::translate("chat", "Dialog", nullptr));
        words4->setText(QCoreApplication::translate("chat", "OI!", nullptr));
        words1->setText(QCoreApplication::translate("chat", "hello", nullptr));
        words3->setText(QCoreApplication::translate("chat", "meowwwwww", nullptr));
        words2->setText(QCoreApplication::translate("chat", "goodbye", nullptr));
        refreshbtn->setText(QString());
        responsetext->setText(QCoreApplication::translate("chat", "meowwww~~~", nullptr));
        Cat->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class chat: public Ui_chat {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CHAT_H
