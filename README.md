# Image Processing & Computer Vision - 3-Hour Workshop

## 📚 About This Workshop

A comprehensive 3-hour hands-on workshop that takes you from the fundamentals of image processing to advanced deep learning techniques for object detection. This seminar progressively builds your understanding through practical examples and implementations using OpenCV and Python, providing a complete journey through the computer vision pipeline.

**Instructor:** Lior Shilon

## 🎯 Learning Objectives

By the end of this workshop, students will:
- ✅ Understand digital image fundamentals and pixel manipulation
- ✅ Master essential image processing techniques (transformations, filtering, edge detection)
- ✅ Extract meaningful features using HOG, GLCM, and LBP methods
- ✅ Understand the complete machine learning pipeline for computer vision

## 🛠️ Prerequisites & Setup

### Required Software
- **VSCode** - Development environment
- **Python 3.10** - Programming language  
- **Anaconda** - Package and environment management

### Environment Setup

#### Step 1: Install Anaconda
Follow the installation guide in the seminar materials

#### Step 2: Create Virtual Environment
```bash
conda create -n cv_seminar_env python=3.10 -y
conda activate cv_seminar_env
```

#### Step 3: Install Required Libraries
```bash
pip install opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84 \
            numpy==1.26.4 matplotlib==3.9.2 scipy==1.13.1 \
            tensorflow==2.17.0 scikit-image==0.24.0 scikit-learn==1.5.2
```

## 📂 Repository Contents

```
├── 📄 seminar_slides.pdf                    # Complete presentation slides
├── 📓 Part1_foundation.ipynb               # Core concepts and basic operations
├── 📓 Part2_image_processing_techniques.ipynb  # Advanced processing techniques  
├── 📓 Part3_feature_extraction.ipynb       # Feature extraction methods
├── 📓 Part3_feature_selection.ipynb        # Feature selection strategies
└── 📝 README.md                             # This file
```

## 📖 Workshop Outline

### Part 1: Foundation - Core Concepts of Image Processing

#### Digital Image Fundamentals
- Understanding pixels as building blocks
- Grayscale (0-255) vs Color (RGB/BGR) representation
- Image coordinate system (origin at top-left)
- Array representation and NumPy basics

#### Basic OpenCV Operations
- Loading and exploring images (`cv2.imread()`)
- Displaying images (`cv2.imshow()`, `cv2.waitKey()`)
- Accessing and manipulating individual pixels
- Drawing geometric shapes (lines, rectangles, circles)

### Part 2: Image Processing Techniques

#### Image Transformations
- **Resizing** - Maintaining aspect ratios
- **Translation** - Moving images in x/y directions
- **Rotation** - Using rotation matrices
- **Flipping** - Horizontal/vertical mirroring
- **Cropping** - Array slicing techniques

#### Arithmetic & Bitwise Operations
- **Addition/Subtraction** - With saturation handling
- **Bitwise Operations** - AND, OR, XOR, NOT
- **Masking** - Selective region processing

#### Filtering & Blurring
- **Mean Filtering** - Simple averaging
- **Gaussian Filtering** - Natural blur with normal distribution
- **Median Filtering** - Salt-and-pepper noise removal
- **Bilateral Filtering** - Edge-preserving smoothing

#### Binarization & Thresholding
- **Simple Thresholding** - Manual threshold selection
- **Adaptive Thresholding** - Local neighborhood-based
- **Otsu's Method** - Automatic optimal threshold

#### Edge Detection & Contours
- **Gradient-Based Methods** - Sobel and Laplacian derivatives
- **Canny Edge Detection** - Multi-stage edge detection
- **Contour Detection** - Shape analysis and object detection

#### Morphological Transformations
- **Basic Operations** - Erosion, Dilation
- **Compound Operations** - Opening, Closing
- **Advanced Operations** - Gradient, Top Hat, Black Hat

#### Template Matching
- Pattern recognition in larger images
- Similarity measurement techniques

### Part 3: Feature Extraction & Machine Learning Pipeline

#### Feature Extraction Methods
- **Color Histograms** - Color distribution analysis
- **GLCM** - Texture analysis through co-occurrence matrices
- **HOG** - Gradient orientation histograms for shape detection
- **LBP** - Local texture patterns

#### Feature Selection Strategies
- **Filter Methods** - Statistical correlation tests
- **Wrapper Methods** - Forward/backward selection
- **Embedded Methods** - L1/L2 regularization

#### Machine Learning Pipeline
- **Supervised Learning** - Training with labeled data
- **Unsupervised Learning** - Pattern discovery without labels
- **Model Training** - From features to predictions
- **Model Deployment** - Production architectures

## 🚀 Getting Started

1. **Clone the repository** to your local machine
2. **Set up the Anaconda environment** following the setup instructions
3. **Open VSCode** and select the cv_seminar_env interpreter
4. **Start with Part1_foundation.ipynb** and progress sequentially
5. **Reference the PDF** for detailed explanations and theory

## ⚠️ Common Pitfalls to Avoid

- 🔴 OpenCV uses **BGR** not RGB format
- 🔴 Confusing (x,y) coordinates with [row, column] indexing
- 🔴 Forgetting `cv2.waitKey()` after `cv2.imshow()`
- 🔴 Not calling `cv2.destroyAllWindows()` to clean up

## 💡 Key Takeaways

### Image Fundamentals
- Images are 2D arrays of pixel values
- Coordinate system: origin (0,0) at top-left
- Access pixels: `image[y, x]` or `image[row, column]`

### Processing Pipeline
1. **Image Ingestion** → Capture and digitize
2. **Preprocessing** → Standardize and enhance
3. **Feature Extraction** → Extract meaningful patterns
4. **Feature Selection** → Choose relevant features
5. **Model Training** → Learn from patterns
6. **Prediction** → Apply to new images

## 📚 Additional Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Scikit-image Documentation](https://scikit-image.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## 🤝 Support

For questions about the workshop content, please refer to the detailed explanations in the PDF presentation or consult the documentation links provided above.

---

**Happy Learning!** Transform pixels into predictions with computer vision! 🖼️→🤖