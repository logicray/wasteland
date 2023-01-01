#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#Author logic-pw
#mangobada@163.com 

import string
import re

class segment():
    def __init__(self, dict_file='lexicon.txt'):
        self.__dict = {}
        self.dict_file=dict_file
    #load dict, key is the first charcter, value is full word 
        words = [line.strip() for line in open(self.dict_file)]
        for word in words:
            first_char = word[0]
            self.__dict.setdefault(first_char, [])
            self.__dict[first_char].append(word)      
        #sort by the length of the word
        for first_char, words in self.__dict.items():
            self.__dict[first_char] = sorted(words, key=lambda x:len(x), reverse=True)

    #match non-chinese character
    def __match_ascii(self, i, input_text):
        #match continue ascii letters
        result = ''
        for i in range(i, len(input_text)):
            if not input_text[i] in string.ascii_letters: 
                break
            result += input_text[i]
        return result

    #match word from location of cursor
    def __match_word(self, first_char, cursor , input_text):
        key=self.__dict.get(first_char)
        if not key:
            if first_char in string.ascii_letters:
                return self.__match_ascii(cursor, input_text)
            return first_char
        words = self.__dict[first_char]
        for word in words:
            if input_text[cursor:cursor+len(word)] == word:
                return word
        return first_char

    #tokenize the input，
    def __tokenize(self, input_text):
        if not input_text:
            return []
        tokens = []
        cursor = 0
        while cursor < len(input_text):
            first_char = input_text[cursor]
            matched_word = self.__match_word(first_char, cursor, input_text)
            tokens.append(matched_word)
            cursor += len(matched_word)
        return tokens 

    #remove punctuation and empty character
    def __cleaning(self, words_list):
        result=[]
        for word in words_list:
            word = re.sub(r"[\s+\.\!\/_,$%^*()+\"\':]+|[+——！，。？、~@#￥%……&*（）：]+", "",word)
            result.append(word)
        result=[r for r in result if len(r)!=0]
        return result

    #
    def parse(self,text):
        try:
            tokens=self.__tokenize(text)
        except Exception as e:
            print(e)
        return self.__cleaning(tokens)

'''
#
def tokenize_test(text):
    load_dict()
    tokens = tokenize(text)
    for token in tokens:
        print(token,end='\t')
        print()

if __name__ == '__main__':
    #load_dict_test() 
    tokenize_test(u'美丽的花园里有各种各样的小动物')
'''
