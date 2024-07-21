# ML_NLP-TF
This repository contains two main components:  TensorFlow Image Prediction Model: A neural network built using TensorFlow that predicts classes of images and evaluates the model's accuracy. NLP Models Using BERT and Transformers: Various Natural Language Processing (NLP) tasks performed using the BERT model and the transformers library.

#TensorFlow Image Prediction Model
Model Architecture
The image prediction model is built using TensorFlow's Keras API. It uses a sequential model with the following layers:

1. Flatten layer to convert input images from 2D to 1D.
2. Dense layers with ReLU activation.
3. Final dense layer for classification.

#NLP Models Using BERT and Transformers
It also includes various NLP tasks using the BERT model and the transformers library.

1. Text Classification: Classifying whole sentences into predefined categories.

2. Token Classification: Classifying each word in a sentence, useful for tasks like Named Entity Recognition (NER).

3. Question Answering: Answering questions based on a given context using BERT's question-answering pipeline.

4. Text Summarization: Summarizing long texts into concise summaries using transformers.

6. Fill in the Blanks: Using masked language models to fill in missing words in sentences.

7. Translation: Translating text from one language to another using transformers.

8. Sample Reviews: A curated dataset showcasing diverse review texts.

9. Sentiment Labels: Each review is labeled as 0 for negative sentiment and 1 for positive sentiment.

10. Code Examples: Implementations demonstrating how to preprocess text data, train sentiment analysis models, and evaluate their performance.

#Yolo for Object detection in images,

1. Demonstrates how to perform image object detection using YOLO (You Only Look Once) versions 3 and 4.
2. Interactive Jupyter notebooks to visualize detections and understand the model's performance.
3. Below link of "AlexeyAB" github repo provide the files such as class names, pre-trained weights, configuration files, for easy setup of yolo version 3 and 4. You can find all the required files in the code from below repo,
   https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3.cfg
4. Inference: Load an image and perform object detection using the YOLOv4 model.
5. Visualization: Visualize the detected objects with bounding boxes and class labels.
   
#Tip: While running the Google Colab notebook (https://colab.research.google.com/), make sure to select the runtime as T4 GPU. This will significantly speed up cell execution. If you don't select the T4 GPU, the process will take a lot more time to complete.

Additionally, you can check out this cool article written by Maneesh Chaturvedi on Medium. Here's a link to it to get you a basic understanding.
(https://maneesh-chaturvedi.medium.com/the-age-of-transformers-3ecbd660892c)
