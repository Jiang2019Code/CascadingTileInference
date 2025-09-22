#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import multiprocessing


def getBatchList(inputList, batchSize):
    listLen = len(inputList)
    batchList = []
    a, b = divmod(listLen, batchSize)
    if b == 0:
        for i in range(a):
            batchList.append(inputList[i * batchSize:(i + 1) * batchSize])
    else:
        for i in range(a):
            batchList.append(inputList[i * batchSize:(i + 1) * batchSize])
        batchList.append(inputList[a * batchSize:])
    return batchList


def parrallCompute(parallCalculateFunction, argsTupleList):
    q = multiprocessing.Queue()
    jobs = []

    for i in range(len(argsTupleList)):
        p = multiprocessing.Process(name=str(i), target=parallCalculateFunction,
                                    args=(argsTupleList[i]))
        jobs.append(p)
        p.start()
        # p.join()
        q.put(i)
    resultList = [q.get() for j in jobs]
    return resultList


def asyncParrallCompute(parrallCalculateFunction, argsTupleList, p, callBack=None):
    q = multiprocessing.Queue()
    jobs = []
    for i in range(len(argsTupleList)):
        p = multiprocessing.Process(target=parrallCalculateFunction, args=(argsTupleList[i] + [q]))
        jobs.append(p)
        p.start()
    resultList = [q.get() for j in jobs]

    for p in jobs:
        p.join()

    return resultList
