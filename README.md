# Graph Attention Network (GAT) for Chest X-Ray Image Classification

## Project Overview
This project applies a **Graph Attention Network (GAT)** to the classification of Chest X-Ray images. The GAT is a type of Graph Neural Network (GNN) that employs attention mechanisms to process graph-structured data. The network dynamically learns which neighboring nodes (in this case, images) are most relevant for aggregating information. This makes it particularly effective for node classification tasks such as the one presented in this project, where the goal is to classify X-ray images based on extracted features and similarities between images.

### Key Concepts:
- **Attention Mechanism**: Nodes in the graph apply normalized attention scores to their neighboring nodes, allowing the model to focus on more important relationships.
- **Multi-head Attention**: Multiple attention heads are used to capture different aspects of node relationships, improving the model's capacity to represent complex patterns.
- **Image as Nodes, Similarities as Edges**: Each X-ray image is represented as a node in the graph. Edges between nodes are established based on cosine similarity between image features.

## Data Preprocessing

1. **Feature Extraction**:
   - The **VGG16** model is used to extract features from the Chest X-ray images. This step transforms raw images into more informative and abstract representations.
   
2. **Dataset Creation**:
   - The extracted features, along with image paths and corresponding labels, are combined into a structured dataset. This dataset forms the basis for the graph construction.

3. **Similarity Determination**:
   - Using **cosine similarity**, the features of images are compared to determine their similarity. Images with high similarity are linked in the graph through edges.

4. **Edge Creation**:
   - Based on cosine similarity scores, edges are created between similar images, allowing the GAT to consider relationships between similar nodes (images).

## Model Architecture

The **Graph Attention Network (GAT)** is built with multiple layers to aggregate information from neighboring nodes in an N-hop fashion, where N corresponds to the number of layers. The attention mechanism allows the model to weigh the importance of each neighbor's contribution when updating node representations.

### Key stages in the GAT model:
- **Message Passing**: Each node aggregates information from its neighbors.
- **Attention Mechanism**: Attention coefficients determine the importance of each neighboring node.
- **Feature Transformation**: Before aggregation, neighbor features are transformed for compatibility.
- **Aggregation and Update**: Neighboring features are aggregated and used to update node features.

## Performance
- **Training and Validation Loss/Accuracy**: The model exhibits stable learning and generalization, with training and validation metrics improving across epochs. It is recommended to train for more than 30 epochs for optimal results.
- **ROC Curve**: The model demonstrates strong classification performance, as shown by the Receiver Operating Characteristic (ROC) curve.
- **Confusion Matrix and Classification Report**: A detailed evaluation of the GAT model's performance is provided, showing how well it handles different classes of X-ray images.

## Results and Evaluation
- The **GAT model** provides a competitive solution for chest X-ray classification tasks, achieving strong results in accuracy, loss, and area under the curve (AUC). 
- The **Confusion Matrix** and **Classification Report** indicate that the GAT model effectively distinguishes between classes, although further training may yield even better results.

## Future Improvements
- Running more than **30 epochs** for training could lead to further improvements in model performance.
- Additional graph-based data preprocessing techniques, as well as experimentation with deeper GAT architectures, could be explored.

## How to Run
1. **Data Preparation**: Ensure that the chest X-ray dataset is preprocessed using VGG16 for feature extraction, and cosine similarity is computed for edge creation.
2. **Install Dependencies**: Install the necessary Python libraries by running:
   ```bash
   pip install -r requirements.txt
