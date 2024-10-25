# Self-Supervised Video Representation Learning via Space-Time Graph Random Walks

This repository contains the implementation of my Master's thesis project in Data Science at Sapienza University of Rome. For a complete description of the methodology and results, please refer to the thesis document: `master-thesis-final.pdf`.

## Overview
This project introduces a novel approach to self-supervised representation learning from video using space-time graphs and superpixel segmentation. The method models video as a graph where nodes represent objects in frames, connected by edges that define temporal correspondence between adjacent frames.

## Key Features
- **Space-Time Graph Modeling**: Videos are represented as graphs with objects as nodes and temporal connections as edges
- **Self-Supervised Learning**: Uses palindrome sequences to exploit cycle-consistency in time
- **Superpixel Segmentation**: Implements unsupervised object extraction through superpixel algorithms
- **Efficient Training**: Achieves up to 5x reduction in training time through reduced image primitives
- **Video Object Segmentation**: Evaluates learned representations through label propagation tasks

## Technical Approach

### Graph Construction
- Nodes: Objects extracted via superpixel segmentation
- Edges: Connect objects between adjacent frames
- Edge Weights: Represent pairwise similarity between nodes
- Random Walk: Defines transition probabilities between nodes

### Training Process
1. Generate palindrome sequences for self-supervision
2. Maximize likelihood of returning to initial node after random walk
3. Utilize cycle-consistency in time for learning
4. Combine patches and superpixels for optimal performance

### Performance Improvements
- 30% reduction in training time
- Larger batch size support
- Faster loss convergence
- Comparable accuracy to baseline models

## Results
The model achieves performance nearly equivalent to previous approaches while significantly reducing computational requirements:
- Training time reduced by >30%
- Maintains competitive accuracy on video object segmentation tasks
- Demonstrates feasibility of superpixel-based approach

## Future Work
Ongoing research areas include:
- Further optimization of object-based space-time graph modeling
- Enhancement of superpixel segmentation techniques
- Investigation of additional self-supervision strategies
- Scaling to larger datasets and longer sequences
