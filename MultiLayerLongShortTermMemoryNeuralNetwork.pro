QT -= core gui

TARGET = LongShortTermMemoryNeuralNetwork
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    io.cpp \
    text.cpp \
    lstmlayer.cpp \
    lstmlayerstate.cpp

HEADERS += \
    io.h \
    text.h \
    lstmlayer.h \
    lstmlayerstate.h

