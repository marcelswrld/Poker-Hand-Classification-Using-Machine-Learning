# Poker-Hand-Classification-Using-Machine-Learning

## Overview
This project involves preprocessing poker hand data to train a Support Vector Classifier (SVC) that learns to differentiate between various poker hands. It was developed as a group assignment for PIC 16A at UCLA, focusing on practical applications of machine learning techniques.

## Project Details

### Data Representation
A standard deck of 52 cards is encoded as follows in the data arrays:
- `0 to 12`: A, 2, 3, ..., 10, J, Q, K of the first suit
- `13 to 25`: A, 2, 3, ..., 10, J, Q, K of the second suit
- `26 to 38`: A, 2, 3, ..., 10, J, Q, K of the third suit
- `39 to 51`: A, 2, 3, ..., 10, J, Q, K of the fourth suit

### Example Encodings
- `np.array([2, 13, 26, 28, 41])`: Represents 3, A, A, 3, 3 of various suits (a full house)
- `np.array([1, 14, 20, 27, 40])`: Represents 2, 2, 8, 2, 2 of various suits (four of a kind)

### Objective
The goal is to preprocess the data effectively so that the SVC can achieve high accuracy in classifying poker hands into categories such as full houses, straight flushes, pairs, etc.

### Challenges and Solutions
Initially, the classifiers (implemented in `gp_checker1.py` and `gp_checker2.py`) performed poorly, with accuracies around 70% and 60%, respectively. By refining the preprocessing steps in `gp.py`, we aimed to improve the classifier performance close to 100% accuracy, especially for complex categories.

### Training and Testing Setup
The dataset was divided into training and testing sets using the partition function which ensures a random but structured split, ensuring that testing data is not included in the training set.

#### Data Split Examples:
- **Full Houses**: Trained on 28 samples, tested on 50 samples
- **Four of a Kinds**: Trained on 28 samples, tested on 50 samples
- **Straight Flushes**: Trained on 8 samples, tested on 20 samples

In `gp_checker2.py`, the training setup was more extensive, including larger sets for straights and flushes (trained on 100, tested on 800).

### Implementation Notes
- The `partition` function plays a crucial role in ensuring the integrity and randomness of training and testing data splits.
- List comprehensions were heavily used to generate training and testing arrays efficiently.

## Conclusion
This project not only challenged us to apply machine learning principles effectively but also to work collaboratively in a group setting to solve a complex problem. The insights gained from tuning the SVC and managing data intricacies were invaluable.
