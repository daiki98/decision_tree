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
   
