# -*- coding:utf-8 -*-
# !/usr/bin/python

import logging
import sys


def remove_title():
    with open('./tangshi.txt') as f:
        for line in f:
            if not ('卷' in line and '_' in line):
                with open('new.txt', 'a') as f2:
                    f2.write(line)
                # print line


def one_sentence_each_line():
    with open('new.txt', 'r') as f:
        for line in f:
            if '，' in line or '。' in line:
                with open('new2.txt', 'a') as f2:
                    line2 = line.replace('，', '\n').replace('。', '\n')
                    f2.write(line2)


def main():
    reload(sys)
    sys.setdefaultencoding('utf-8')

    try:
        # remove_title()
        one_sentence_each_line()
    except Exception as e:
        print(e)
        logging.exception(e)
        logging.exception('')
        sys.exit(2)


if __name__ == "__main__":
    main()
