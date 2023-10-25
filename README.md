# 04-Types-of-Flower-Classification-with-TPUs

Description:

This project leverages Tensor Processing Units (TPUs) for the classification of 104 types of flowers. TPUs are specialized hardware accelerators designed for deep learning tasks and were initially developed by Google. They offer exceptional speed and performance for processing large image datasets, making them an excellent choice for this flower classification task.

The task is a great opportunity to explore the capabilities of TPUs and harness the power of advanced machine learning techniques, especially with the support of the latest Tensorflow release (TF 2.1). TF 2.1 focuses on TPUs and provides support both through the Keras high-level API and at a lower level, allowing for custom training loops.

Challenge Overview:

Our natural world is incredibly diverse and vast. There are over 5,000 species of mammals, 10,000 species of birds, 30,000 species of fish, and astonishingly, over 400,000 different types of flowers. This task  is specifically focused on classifying 104 types of flowers based on their images. These images are drawn from five different public datasets, and the classes exhibit varying degrees of specificity. Some classes are very narrow, containing only a particular sub-type of flower (e.g., pink primroses), while other classes contain many sub-types (e.g., wild roses).

Dataset Description:

The dataset for this task contains images in the TFRecord format. TFRecord is a container format commonly used in TensorFlow for grouping and sharding data files, which optimizes training performance. Each TFRecord file includes the following information:

id: A unique identifier for each sample.

label: The class of the sample for training data (the type of flower).

img: The actual image data, represented in an array format.

The dataset is organized into the following categories:

train/*.tfrec: Training samples, including labels.

val/*.tfrec: Pre-split training samples with labels to help with model performance checking on TPU. The split is stratified across labels.

test/*.tfrec: Samples without labels. Participants will be predicting the classes of flowers for these images.
