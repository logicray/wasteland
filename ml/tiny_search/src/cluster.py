#!/usr/bin/python3
#logic-pw
#2015.12.01
#mangobada@163.com

from math import sqrt
import crawler
import matplotlib.pyplot as plt
import random
import pymongo

def load_data(filename):
    lines=[line for line in open(filename)]

    colnames=lines[0].strip().split('\t')[1:]
    rownames=[]
    data=[]
    for line in lines[1:]:
        p=line.strip().split('\t')
        rownames.append(p[0])
        data.append([float(x) for x in p[1:]])
    return rownames,colnames,data


def pearson(v1,v2):
    sum1=sum(v1)
    sum2=sum(v2)

    sum1Sq=sum([pow(v,2) for v in v1])
    sum2Sq=sum([pow(v,2) for v in v2])

    pSum=sum([v1[i]*v2[i] for i in range(len(v1))])

    num=pSum-(sum1*sum2/len(v1))
    den=sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
    if den==0:
        return 0

    return 1.0 - num/den

class bicluster:
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None):
        self.left=left
        self.right=right
        self.vec=vec
        self.id=id
        self.distance=distance

    def hcluster(rows,distance=pearson):
        distances={}
        currentclustid=-1
        clust=[bicluster(rows[i],id=i) for i in range(len(rows))]
        while len(clust)>1:
            lowestpair=(0,1)
            closet=distance(clust[0].vec,clust[1].vec)

        for i in range(len(clust)):
            for j in range(i+1,len(clust)):
                #distance is saved
                if (clust[i].id,clust[j].id) not in distances:
                    distances[(clust[i].id,clust[j].id)]=distance(clust[i].vec,clust[j].vec)
                d=distances[(clust[i].id,clust[j].id)]
                if d<closet:
                    closest=d
                    lowestpair=(i,j)
        #print(clust[0].vec)
        print(len(clust[0].vec))
        print(lowestpair)
        mergevec=[(clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]

        newcluster=bicluster(mergevec, left=clust[lowestpair[0]],
                right=clust[lowestpair[1]],
                distance=closest,id=currentclustid)
        #cluster ids that weren't in the original set are negative
        currentclustid-=1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)
    return clust[0]

    def printclust(clust,labels=None,n=0):
    #indent to make a hierarchy layout
    for i in range(n):
        print(' ',end='')
    if clust.id<0:
        #negative id means that this is branch
        print('-')
    else:
        #positive id means that this is an endpoint
        if labels==None:
            print(clust.id)
        else:
            print(labels[clust.id])
    if clust.left!=None:
        printclust(clust.left,labels=labels,n=n+1)
    if clust.right!=None:
        printclust(clust.right,labels=labels,n=n+1)


def getheight(clust):
    if clust.left==None and clust.right==None:
        return 1
    return getheight(clust.left)+getheight(clust.right)

def getdepth(clust):
    if clust.left==None and clust.right==None:
        return 0
    return max(getdepth(clust.left),getdepth(clust.right))+clust.distance


def rotatematrix(data):
    newdata=[]
    for i in range(len(data[0])):
        newrow=[data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata

def kcluster(rows,distance=pearson,k=4):
    ranges=[(min([row[i] for row in rows]),max([row[i] for row in rows])) for i in range(len(rows[0]))]
    clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0] for i in range(len(rows[0]))] for j in range(k)]
    lastmatches=None
    for t in range(100):
        print('Iteration %d' %t)
        bestmatches=[ [] for i in range(k)]
        for j in range(len(rows)):
            row=rows[j]
            bestmatch=0
            for i in range(k):
                d=distance(clusters[i],row)
                if d<distance(clusters[bestmatch],row):
                    bestmatch=i
            bestmatches[bestmatch].append(j)
        if bestmatches==lastmatches:break
        lastmatches=bestmatches
        for i in range(k):
            avgs=[0.0]*len(rows[0])
            if len(bestmatches[i]) >0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m]+=rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j]/=len(bestmatches[i])
                clusters[i]=avgs
    return bestmatches
