# Decision Tree Algorithm
* A decision tree is flowchart-like tree structure.
* Decision Tree shares internal decision-making logic(white box type).
* Training time is faster than the neural network algorithm.
* The decision tree is a <strong>distribution-free</strong> or <strong>non-parametric</strong> method, which does not depend upon probability distribution assumptions.
* Decision trees can handle high dimensional data with good accuracy.


# How does the Decision Tree work?
1. Select the best attribute using Attribute Selection Measures(ASM) to split the records.
2. Make that attribute a decision node and breaks the dataset into smaller subsets.
3. Starts tree building by repeating this process recursively for each child until one of the condition will match:
    * All the tuples belong to the same attribute value.
    * There are no more remaining attributes.
    * There are no more instances.
   
# Attribute Selection Measures
Attribute Selection Measure is a heuristic for selecting the splitting criterion that partition data into the best possible manner.
It's also known as splitting rules because it helps us to determine breakpoints for tuples on a given node.
Best score attribute will be selected as a splitting attribute.<br>

Most popular selection measures are following three methods:
   * Information Gain
   * Gain Ratio
   * Gini Index

# What is seaborn?
* <a href="https://leaf-organ-558.notion.site/seaborn-heatmap-ba13f6064d6f4bd9afee54cf73ffed2f">note</a>

# Gini Impurity
* <a href="https://leaf-organ-558.notion.site/Gini-Impurity-c65f1286369f4775b4cd01a11dc3e524">note</a>



   
