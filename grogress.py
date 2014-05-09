#!/usr/bin/env python3

_column = 0


def begin_progress(msg):
    global _column
    if type(msg) != str:
        msg = str(msg)
    _column = len(msg) + 3
    print(msg + '...', end='', flush=True)


def progress():
    global _column
    msg = '.'
    if _column >= 80:
        msg = '\n.'
        _column = 0

    print(msg, end='', flush=True)
    _column += 1


def end_progress():
    global _column
    msg = 'Done.'
    if _column >= 80:
        msg = '\nDone.'
    print(msg)