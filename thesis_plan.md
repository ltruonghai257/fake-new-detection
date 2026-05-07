
# Thesis Plan: Multimodal Misinformation Detection

This document outlines a plan for a thesis on the topic of multimodal misinformation detection, with a focus on extending the ViFactCheck dataset and building a baseline model using contrastive learning.

## Part 1: Dataset Extension

The goal of this part is to create a comprehensive multimodal dataset for misinformation detection by extending the existing ViFactCheck dataset.

### Step 1.1: Data Acquisition & Initial Analysis

*   **Action:** Obtain the ViFactCheck dataset.
*   **Tasks:**
    *   Analyze the existing data structure, including the format of the text, images, and labels.
    *   Identify the limitations of the dataset (e.g., size, scope, lack of multimodality).
    *   Document the statistics of the dataset (number of samples, class distribution, etc.).

### Step 1.2: Data Crawling & Collection

*   **Action:** Collect new data to supplement the ViFactCheck dataset.
*   **Tasks:**
    *   Identify reliable sources for both real and fake news (e.g., reputable news websites, fact-checking organizations, and known sources of misinformation).
    *   Develop web crawlers to automatically collect articles, including text, images, and metadata.
    *   Consider using APIs for social media platforms (like Twitter or Facebook) to gather multimodal posts.

### Step 1.3: Data Cleaning & Preprocessing

*   **Action:** Clean and preprocess the collected data.
*   **Tasks:**
    *   **Text:**
        *   Remove HTML tags, boilerplate content, and advertisements.
        *   Standardize the text format (e.g., unicode normalization).
        *   Perform sentence and word tokenization.
    *   **Images:**
        *   Filter out low-quality or irrelevant images (e.g., logos, ads).
        *   Resize and normalize images to a consistent format.
        *   Handle missing images.

### Step 1.4: Data Annotation & Labeling

*   **Action:** Label the newly collected data.
*   **Tasks:**
    *   Develop a clear and consistent annotation guideline for labeling news as "real" or "fake".
    *   Perform manual annotation of the collected data. This is a critical and time-consuming step.
    *   Consider a two-step verification process where multiple annotators label the same data to ensure quality.

### Step 1.5: Final Dataset Construction

*   **Action:** Combine the original and new data to create the final dataset.
*   **Tasks:**
    *   Merge the extended data with the ViFactCheck dataset.
    *   Split the final dataset into training, validation, and testing sets.
    *   Document the final dataset, including its size, structure, and statistics.

## Part 2: Baseline Model Development

The goal of this part is to build a baseline model for multimodal misinformation detection using contrastive learning to effectively learn from both text and images.

### Step 2.1: Model Architecture Definition

*   **Action:** Define the architecture of the multimodal model.
*   **Components:**
    *   **Text Encoder:** A transformer-based model to encode the text data. For Vietnamese, a model like PhoBERT would be a good choice.
    *   **Image Encoder:** A vision model to encode the image data. You can use a ResNet-based architecture or a Vision Transformer (ViT). The `SimCLR` model you created is a good starting point for the image encoder.
    *   **Fusion Layer:** A mechanism to combine the representations from the text and image encoders. This could be a simple concatenation followed by a linear layer, or a more complex attention-based fusion method.

### Step 2.2: Contrastive Pre-training

*   **Action:** Pre-train the text and image encoders using a contrastive learning approach.
*   **Tasks:**
    *   **Positive Pairs:** Create "positive" pairs of (text, image) that come from the same news article or social media post.
    *   **Negative Pairs:** Create "negative" pairs by pairing the text from one article with an image from a different article.
    *   **Loss Function:** Use a contrastive loss function (like the `NTXentLoss` in your `simclr.py` file) to train the encoders. The goal is to pull the representations of positive pairs closer together and push the representations of negative pairs further apart.
    *   This pre-training step helps the model learn the semantic relationship between text and images before it even sees the "real" or "fake" labels.

### Step 2.3: Downstream Task - Misinformation Classification

*   **Action:** Fine-tune the model for the misinformation detection task.
*   **Tasks:**
    *   Add a classification head to the fused output of the pre-trained encoders. This head will typically be a linear layer with a softmax activation function to output the probabilities for "real" and "fake".
    *   Train the entire model (or just the classification head) on the labeled dataset you created in Part 1.

### Step 2.4: Evaluation

*   **Action:** Evaluate the performance of the baseline model.
*   **Tasks:**
    *   **Metrics:** Use standard classification metrics to evaluate your model, such as:
        *   Accuracy
        *   Precision
        *   Recall
        *   F1-score
    *   **Analysis:**
        *   Perform an error analysis to understand the cases where the model fails.
        *   Compare the performance of the multimodal model with text-only and image-only baselines to demonstrate the benefit of using multiple modalities.

## Timeline

| Phase                  | Task                               | Estimated Time |
| ---------------------- | ---------------------------------- | -------------- |
| **Part 1: Dataset**    |                                    | **(Weeks 1-6)**|
|                        | 1.1: Acquisition & Analysis        | 1 week         |
|                        | 1.2: Crawling & Collection         | 2 weeks        |
|                        | 1.3: Cleaning & Preprocessing      | 1 week         |
|                        | 1.4: Annotation & Labeling         | 2 weeks        |
| **Part 2: Model**      |                                    | **(Weeks 7-12)**|
|                        | 2.1: Architecture Definition       | 1 week         |
|                        | 2.2: Contrastive Pre-training      | 2 weeks        |
|                        | 2.3: Fine-tuning for Classification| 2 weeks        |
|                        | 2.4: Evaluation & Analysis         | 1 week         |
| **Thesis Writing**     |                                    | **(Weeks 13-16)**|
|                        | Writing, revisions, and submission | 4 weeks        |

## Tools and Technologies

*   **Programming Language:** Python
*   **Deep Learning Framework:** PyTorch or TensorFlow
*   **Text Processing:** Hugging Face Transformers (for models like PhoBERT), NLTK
*   **Image Processing:** OpenCV, Pillow, Torchvision
*   **Web Crawling:** Scrapy, BeautifulSoup
*   **Data Manipulation:** Pandas, NumPy
