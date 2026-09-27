"""
Backgammon with Expectiminimax AI - Entry point
Name:       Krishna Mahajan
Student ID: 106wucww
Module:     5ENT1140 Artificial Intelligence Principles - Mini Project

pygbag (the browser publisher) requires the entry file to be called main.py.
Run on desktop:  python3 main.py
Build for web:   pygbag .
"""

import asyncio
import pygame   # must be imported here: pygbag scans main.py to decide which libraries to load
from gui import BackgammonApp


async def main():
    """Creates the game window and runs the async main loop."""
    await BackgammonApp().run()


asyncio.run(main())
