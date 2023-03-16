#Mac OS X, Python3.5
#_*_encoding=utf-8_*_
#Author@logic-pw
#mangobada@163.com

'''
a self defined crawler and data is 
stored in mongoDB 
'''
import re
import urllib.request
import os
import queue
import pymongo
import segment

from datetime import datetime
from bs4 import BeautifulSoup
from io import BytesIO
from  gzip import GzipFile
from urllib.parse import urljoin

#class my_crawler():
start_urls="http://news.163.com"
#download the webpage of the url
def dnload(url):
    req = urllib.request.urlopen(url)
    #uncompress gzip
    if req.info().get('Content-Encoding')=='gzip':
        f=BytesIO(req.read())
        req=GzipFile(fileobj=f,mode="rb")
    page = req.read()
    #try:
    #    content_type=req.info().get("Content-Type")
    #except:
    #    content_type="text/html; charset=utf-8"
    #get the codec
    #try:
    #    codec=re.compile(r'charser=(.+)').findall(content_type)[0]
    #except:
    #    codec='utf-8'
    page = page.decode('GBK')
    return page

#get the urls from the page by regular expression
def parser(page):
    pattern=re.compile(r'href="(http://news.163.com[A-Za-z0-9\.\/]*?\.html)"',re.DOTALL)
    match=pattern.findall(page)
    soup=BeautifulSoup(page,'lxml')
    texts=soup('p')
    content=''
    for text in texts:
        if text.string!=None:
            content+=text.string
            content+='\n'
    return match,str(content)

#another parser by BeautifulSoup 
def parser2(page):
    result=set()
    soup=BeautifulSoup(page,"lxml")
    links=soup('a')
    for link in links:
        if ('href' in dict(link.attrs)):
            url=urljoin(page,link['href'])
            if url.find("'")!=-1:
                continue
            url=url.split('#')[0]
            if url[0:4]=='http':
                result.add(url)
    return result

def main():
    #connect to mongoDB
    client=pymongo.MongoClient()
    db=client.mydb
    seen=set()
    url_queue=queue.Queue()
    seen.add(start_urls)
    url_queue.put(start_urls)
    while(url_queue.empty()==False):
        current_url=url_queue.get()
        print(current_url)
        try:
            urlSet,content=parser(dnload(current_url))
            doc={"url":current_url,
                "date":datetime.now(),
                "text":content}
            db.news.insert_one(doc)
        except Exception as e:
            print(e)
            continue
        for url in urlSet:
            if url not in seen:
                seen.add(url)
                url_queue.put(url)
    client.close()

#
if __name__=="__main__":
    main()

#class scrapy_crawler():
import feedparser
import re

def getwordcounts(url):
    wc={}
    #for url in urls:
    #    d=feedparser.parse(url)
    d=feedparser.parse(url)
    words=getwords(d.feed.summary)
    for word in words:
        wc.setdefault(word,0)
        wc[word]+=1
    return url,wc

def getwords(html):
    #remove all the html tag
    txt=re.compile(r'<[^>]+>').sub('',html)
    
    #split words by all non-alpha characters
    words=re.compile(r'[^A-Z^a-z]+').split(txt)
    return [word.lower() for word in words if word!='']

apcount={}
wordcounts={}
for feedurl in open('feedlist.txt'):
    title,wc=getwordcounts(feedurl)
    wordcounts[title]=wc
    for word,count in wc.items():
        apcount.setdefault(word,0)
        if count>1:
            apcount[word]+=1

wordlist=[]
for w,bc in apcount.items():
    frac=float(bc)/len(wordcounts)
    if frac>0.1 and frac<0.5:
        wordlist.append(w)

out=open('blogdata.txt','w')
out.write('Blog')
for word in wordlist:
    out.write('\t%s' %word)
out.write('\n')
for blog,wc in wordcounts.items():
    blog=re.compile(r'[\n]+').sub('',blog)
    out.write('%s' %blog)
    for word in wordlist:
        if word in wc:
            out.write('\t %d' % wc[word])
        else:
            out.write('\t0')
    out.write('\n')

import feedparser
import re

def getwordcounts(url):
    d=feedparser.parse(url)
    wc={}

    for e in d.entries:
        if 'summary' in e:
            summary=e.summary
        else:
            summary=e.description

        words=getwords(e.title+' '+summary)
        for word in words:
            wc.setdefault(word,0)
            wc[word]+=1
        return d.feed.title,wc

def getwords(html):
    txt=re.compile(r'<[^>]+>').sub('',html)

    words=re.compile(r'[^A-Z^a-z]+').split(txt)
    return [word.lower() for word in words if word !='']

apcount={}
wordcounts={}
for feedurl in file('feedlist.txt'):
    title,wc=getwordcounts(feedurl)
    wrdcounts[title]=wc
    for word,count in wc.items():
        apcount.setdefault(word,0)
        if count>1:
            apcount[word]+=1

wordlist=[]
for w,bc in apcount.items():
    frac = float(bc)/len(feedlist)
    if frac>0.1 and frac<0.5:
        wordlist.append(w)

out=open('blogdata.txt','w')
out.write('blog')
for word in wordlist:
    out.write('\t%s' %word)
    out.write('\n')
    for blog,wc in wordcounts.items():
        out.write(blog)
        for word in wordlist:
            if word in wc:
                out.write('\t%d ' %wc[word])
            else:
                out.write('\t0')
        out.write('\n')
