#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv
import math
from heapq import heapify, heappush, heappushpop, nlargest, nsmallest


# ## Construindo Árvore KD

# ### Class Node

# In[2]:


class Node:
    left = None
    right = None
    plane = None
    point = None
    def __init__(self):
        pass
    def __lt__(self, other):
        a = True
        return a
    def __le__(self,other):
        a = False
        return a
    def setDimension(self, dimension):
        self.dimension = dimension
    def setValue(self, value):
        self.value = float(value)


# In[3]:


class KDTree:
    def __init__(self, dataframe):
        self.root = self.BuildKDTree(dataframe, 0)
    def BuildKDTree(self, P, dimension):
        node = Node()
        if len(P) > 1:
            parameters = list(P.columns.values.tolist())[:-1]
            P = P.sort_values(dimension).reset_index(drop=True)
            medianPosition = math.floor(len(P.index) / 2)
            
            node.dimension = dimension
            node.plane = P.iloc[medianPosition][dimension]
            
            P1 = P.iloc[: medianPosition, : ]
            P2 = P.iloc[medianPosition : , : ]
            if len(P1.index) > 0:
                node.left = self.BuildKDTree(P1, (dimension + 1) % len(parameters))
            if len(P2.index) > 0:
                node.right = self.BuildKDTree(P2, (dimension + 1) % len(parameters))
        else:
            node.point = P.values.tolist()[0]
        return node


# In[4]:


class PriorityQueue:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.q = list()
    def put(self, item):
        if len(self.q) < self.maxsize:
            self.q.append(item)
        elif abs(self.top()[0]) > abs(item[0]):
            self.pop()
            self.q.append(item)
    def pop(self):
        if len(self.q) > 0:
            maxPriority = max(self.q, key=lambda item: item[0])
            self.q.remove(maxPriority)
    def top(self):
        return max(self.q, key=lambda item: item[0])
    def bottom(self):
        return min(self.q, key=lambda item: item[0])


# In[5]:


def euclideanDistance(p1, p2):
    distance = 0.0
    for i in range(len(p1) - 1):
        distance = distance + (float(p1[i]) - float(p2[i])) ** 2
    return math.sqrt(distance)
class xNN:
    def __init__(self, train, test, maxsize):
        self.KDTree = KDTree(train)
        self.root = self.KDTree.root
        self.test = test
    def findPriorityQueue(self, priorityQueue, node, point):
        #If node is a leaf
        if node.left == None and node.right == None:
            distance = euclideanDistance(node.point, point)
            priorityQueue.put((distance, node))
        else:
            dimension = node.dimension
        #print("float(point[dimension]) = {} > float(node.value) = {}".format(float(point[dimension]), float(node.value)))
            if float(point[dimension]) > float(node.plane):
                nextBranch = node.right
                oppositeBranch = node.left
            else:
                nextBranch = node.left
                oppositeBranch = node.right
            priorityQueue = self.findPriorityQueue(priorityQueue, nextBranch, point)
            if abs(priorityQueue.top()[0]) > abs(float(point[dimension]) - float(node.plane)):
                priorityQueue = self.findPriorityQueue(priorityQueue, oppositeBranch, point)
        return priorityQueue
    def runTest(self, attributes):
        actualClassification = list()
        predictedClassification = list()
        attributesCount = [0] * len(attributes)
        for i, row in self.test.iterrows():
            priorityQueue = PriorityQueue(5)
            priorityQueue = self.findPriorityQueue(priorityQueue, self.root, row)
            nearestNeighbour = priorityQueue.bottom()[1].point
            actualClassification.append(row.tolist()[-1])
            predictedClassification.append(nearestNeighbour[-1])
            
        actualCategorical = pd.Categorical(actualClassification, categories = attributes)
        predictedCategorical = pd.Categorical(predictedClassification, categories = attributes)
        
        confusionMatrix = pd.crosstab(actualCategorical, predictedCategorical, rownames = ['Actual'], colnames = ['Predicted'], dropna = False)
        print(confusionMatrix)
        print("Acurácia é {}".format(np.diag(confusionMatrix).sum() / confusionMatrix.to_numpy().sum()))
        for i, row in confusionMatrix.iterrows():
            #print("i: {}, row: {}, row[i]: {}".format(i, row, row[i]))
            if row.sum() != 0.0:
                print("Precisão de '{}' é {}".format(i, row[i] / row.sum()))
        for name, data in confusionMatrix.iteritems():
            if data.sum() != 0.0:
                print("Revocação de '{}' é {}".format(name, data[name] / data.sum()))


# In[6]:


def process1(fileName, classification):
        datContent = [i.strip().split() for i in open("./" + fileName + ".dat").readlines()]
        #print(datContent)
        #Write it as a new CSV file
        with open("./" + fileName + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(datContent)

        file = open(fileName + ".csv",'r')
        final_file = open("final_" + fileName + ".csv",'w')
        writer = csv.writer(final_file)
        #Remove .dat headers
        for row in csv.reader(file):
            #If row is not empty
            if(bool(row) != False):
            #If row is not a header
                if(row[0].startswith('@') == False):
                    writer.writerow(row)
            string = " ".join(row)
            #Get attributes list
            if string.startswith('@attribute ' + classification):
                attributes = string.strip('@attribute ' + classification).strip('{}').split(',')
                #print("Attributes: {}".format(attributes))      
        file.close()
        final_file.close()
        #Remove unnecessary commas and quotes
        dataframe = pd.read_csv("./final_" + fileName + ".csv", header = None)
        #print(dataframe)
        dataframe = dataframe.sample(frac = 1).reset_index(drop=True)
        dataframe = dataframe.replace({',':''}, regex=True)
        dataframe = dataframe.replace({'"':''}, regex=True)
        #print(dataframe)
        rows = len(dataframe.index)
        #print(rows)
        train = dataframe.iloc[: math.floor(rows * 0.7), :]
        test = dataframe.iloc[math.floor(rows * 0.7):, :]
        XNN = xNN(train, test, 2)
        XNN.runTest(attributes)


# In[7]:


def process2(fileName, classification):
        datContent = [i.strip().split() for i in open("./" + fileName + ".dat").readlines()]
        #print(datContent)
        #Write it as a new CSV file
        with open("./" + fileName + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(datContent)
        file = open(fileName + ".csv",'r')
        final_file = open("final_" + fileName + ".csv",'w')
        writer = csv.writer(final_file, delimiter = ',', quoting = csv.QUOTE_NONE, escapechar = " ")
        #Remove .dat headers
        for row in csv.reader(file):
            #If row is not empty
            if(bool(row) != False):
            #If row is not a header
                if(row[0].startswith('@') == False):
                    #print(row)
                    row[0] = str(row[0].replace(',', "','"))
                    #print(row)
                    row[0] = str("'" + row[0] + "'")
                    row = row[0].split(",")
                    writer.writerow(row)
            string = " ".join(row)
            #print("String: {}".format(string))
            #Get attributes list
            if string.startswith('@attribute ' + classification):
                attributes = string.strip('@attribute ' + classification).strip('{}').split(',')
                #print("Attributes: {}".format(attributes))      
        file.close()
        final_file.close()
        dataframe = pd.read_csv("./final_" + fileName + ".csv", header = None)
        #print(dataframe)
        dataframe = dataframe.sample(frac = 1).reset_index(drop=True)
        dataframe = dataframe.replace({"'":''}, regex=True)
        dataframe = dataframe.replace({',':''}, regex=True)
        #print(dataframe)
        rows = len(dataframe.index)
        #print(rows)
        train = dataframe.iloc[: math.floor(rows * 0.7), :]
        test = dataframe.iloc[math.floor(rows * 0.7):, :]
        XNN = xNN(train, test, 5)
        XNN.runTest(attributes)


# In[8]:


process1("iris", "Class")


# In[9]:


process2("banana", "Class")


# In[10]:


process2("titanic", "Survived")


# In[11]:


process2("ecoli", "Site")


# In[12]:


process2("phoneme", "Class")


# In[13]:


process2("pima", "Class")


# In[14]:


process1("haberman", "Survival")


# In[15]:


process2("heart", "Class")


# In[16]:


process2("wdbc", "Class")


# In[17]:


process2("appendicitis", "Class")

