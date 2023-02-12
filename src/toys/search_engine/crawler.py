#!/usr/bin/env python3
#Author@logic-pw
#2015.12.10
#mangobada@163.com

"""
this file include two class 
crawler to download,parse pages, and then save to database
"""

from urllib import request, parse
from bs4 import BeautifulSoup

import re
import sqlite3
import pymongo

#the words to ignore
#ignorewords=set(['the','of','to','and','a','in','is','it'])
ignorewords=set(['的'])
class crawler:
    def __init__(self,dbname='mydb'):
        self.client=pymongo.MongoClient("127.0.0.1:27017")
        #self.db=self.client.dbname
        #self.con=sqlite3.connect(dbname)
    def __del__(self): 
        self.client.close()
    def dbcommit(self): 
        self.con.commit()

    # Auxilliary function for getting an entry id and adding
    # it if it's not present
    def getentryid(self,table,field,value,createnew=True):
        cur=self.con.execute("select rowid from %s where %s='%s'" %(table,field,value))
        cur=self.client.dbname.news.find({"%s:}
        res=cur.fetchone()
        if res==None:
            curs=self.con.execute("insert into %s (%s) values ('%s')" % (table,field,value))
            return curs.lastrowid
        else:
            return res[0]

    # Index an individual page
    def addtoindex(self,url,soup):
        if self.isindexed(url):
            return
        print('Indexing %s' %url)

        text=self.gettextonly(soup)
        words=self.separatewords(text)

        urlid=self.getentryid('urllist','url',url)

        for i  in range(len(words)):
            word=words[i]
            if word in ignorewords:
                continue
            wordid=self.getentryid('wordlist','word',word)
            self.con.execute("insert into wordlocation(urlid,wordid,location) \
                    values (%d,%d,%d)" %(urlid,wordid,i))

    # Extract the text from an HTML page (no tags)
    def gettextonly(self,soup):
        v=soup.string
        if v==None:
            c = soup.contents
            resulttext=''
            for t in c:
                subtext=self.gettextonly(t)
                resulttext+=subtext+'\n'
            return resulttext
        else:
            return v.strip()

    # Separate the words by any non-whitespace character
    def separatewords(self,text):
        splitter=re.compile(r'\W*')
        return [s.lower() for s in splitter.split(text) if s!='']

    # Return true if this url is already indexed
    def isindexed(self,url):
        u=self.con.execute("select rowid from urllist where url='%s'" %url).fetchone()
        if u!=None:
            #double check
            v=self.con.execute('select * from wordlocation where urlid=%d' % u[0]).fetchone()
            if v!=None:
                return True
        return False

    # Add a link between two pages
    def addlinkref(self,urlFrom,urlTo,linkText):
        pass
    # Starting with a list of pages, do a breadth
    # first search to the given depth, indexing pages
    # as we go
    def crawl(self,pages,depth=2):
        for i in range(depth):
            newpages=set()
            for page in pages:
                try:
                    c = request.urlopen(page)
                except:
                    print("Could not open %s" %page)
                    continue
                soup=BeautifulSoup(c.read())
                self.addtoindex(page,soup)

                links = soup('a')
                for link in links:
                    if('href' in dict(link.attrs)):
                        url=parse.urljoin(page,link['href'])
                        if url.find("'")!=-1:
                            continue
                        url=url.split('#')[0]
                        if url[0:4]=='http' and not self.isindexed(url):
                            newpages.add(url)
                        linkText=self.gettextonly(link)
                        self.addlinkref(page,url,linkText)
            self.dbcommit()
            pages=newpages

    # Create the database tables
    def createindextables(self):
        self.con.execute('create table urllist(url)')
        self.con.execute('create table wordlist(word)')
        self.con.execute('create table wordlocation(urlid,wordid,location)')
        self.con.execute('create table link(fromid integer,toid integer)')
        self.con.execute('create table linkwords(wordid,linkid)')
        self.con.execute('create index word_index on wordlist(word)')
        self.con.execute('create index url_index on urllist(url)')
        self.con.execute('create index word_url_index on wordlocation(wordid)')
        self.con.execute('create index urlto_index on link(toid)')
        self.con.execute('create index urlfrom_index on link(fromid)')
        self.dbcommit()

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
