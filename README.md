# Chess-Game-Classifier
This machine learning project classifies chess game outcomes (win, lose, or draw) based on various game features using three different classification algorithms: Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM).

# Chess Game Classifier

A machine learning project that classifies chess game outcomes (win, lose, or draw) using Decision Tree, KNN, and SVM models.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project implements three machine learning algorithms to classify chess game outcomes based on various game features. The goal is to compare the performance of Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM) models in predicting game results.

## Features
The classifier analyzes several chess game characteristics including:
- Player ratings (Elo)
- Opening moves
- Piece advantage
- Control of center squares
- King safety metrics
- Pawn structure
- Time remaining
- Number of checks given

## Models
1. **Decision Tree Classifier**
   - Tree-based model with interpretable rules
   - Uses information gain/Gini impurity for splits

2. **K-Nearest Neighbors (KNN)**
   - Instance-based learning
   - Classifies based on similarity to training examples

3. **Support Vector Machine (SVM)**
   - Finds optimal decision boundary
   - Supports linear and non-linear kernels

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/chess-game-classifier.git
   cd chess-game-classifier
