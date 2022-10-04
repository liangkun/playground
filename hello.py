#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import sys
import random
import argparse

def guess_number(options):
    power, min, max, chances = options.power, options.min, options.max, options.chances
    anwser = random.randint(min, max)
    if power:
        anwser = 2 ** anwser

    done = False
    if power:
        print(f"开始guess_nuber游戏，请猜一个在{2**min}和{2**max}之间的2的幂数")
    else:
        print(f"开始guess_number游戏，请猜一个在{min}和{max}之间的数字")
    while not done:
        try:
            guess = int(input(f"\n猜猜是多少，你还有{chances}次机会："))
            if guess > anwser:
                print("太大了，再试一次")
            elif guess < anwser:
                print("太小了，再试一次")
            else:
                print(f"你猜对了，答案是：{anwser}")
                done = True
            chances -= 1
            if chances <= 0:
                if not done:
                    print(f"你输了，答案是：{anwser}")
                    done = True
        except Exception as e:
            pass

REGISTRY = {
    "guess_number": guess_number
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("number games")
    parser.add_argument("--game", type=str, default="guess_number")
    parser.add_argument("--power", default=False, action="store_true")
    parser.add_argument("--min", type=int, default=0)
    parser.add_argument("--max", type=int, default=100)
    parser.add_argument("--chances", type=int, default=8)
    options = parser.parse_args(sys.argv[1:])

    game = options.game
    if game in REGISTRY:
        REGISTRY[game](options)
    else:
        print(f"不知道这个游戏{game}")