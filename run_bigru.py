"""
Entry point: train the BiGRU rumor classification model.

Usage:
    py -3.11 run_bigru.py
"""

from scripts.train_bigru import train

if __name__ == "__main__":
    train()
