"""
Entry point: train the TextCNN + GloVe rumor classification model.

Usage:
    py -3.11 run_textcnn.py
"""

from scripts.train_textcnn import train

if __name__ == "__main__":
    train()
