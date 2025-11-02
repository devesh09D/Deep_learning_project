# Adversarial Patch Attack & Defense System

A comprehensive implementation of adversarial patch attacks against Vision Transformers (ViT) and multi-layered defense mechanisms to counter these attacks.

## 🎯 Project Overview

This project demonstrates the vulnerability of Vision Transformer models to adversarial patch attacks and implements robust defense strategies to mitigate these attacks. The system includes both attack generation and defense evaluation components.

## 📁 Project Structure

```
├── vit.ipynb              # Adversarial patch generation and training
├── eval.ipynb             # Attack evaluation and success rate measurement
├── defense.py             # Complete defense system implementation
├── defense_system.ipynb   # Defense system in notebook format
├── data/                  # Food101 dataset (auto-downloaded)
├── 7's.pt                 # Trained adversarial patch
├── 7.png                  # Patch visualization
├── 7.7.png               # Attack result visualization
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision timm opencv-python matplotlib pillow tqdm scikit-learn numpy
```

### Running the Attack

1. **Generate Adversarial Patch:**
   ```bash
   jupyter notebook vit.ipynb
   ```
   Run all cells to train an adversarial patch that fools ViT models into predicting "spaghetti" (class 957).

2. **Evaluate Attack:**
   ```bash
   jupyter notebook eval.ipynb
   ```
   Test the trained patch on 1000 random Food101 images to measure attack success rate.

3. **Test Defense System:**
   ```bash
   python defense.py
   ```
   Evaluate the multi-layered defense system against your adversarial patches.

## 🔍 Attack Methodology

### Adversarial Patch Generation (`vit.ipynb`)

The attack uses a sophisticated multi-objective optimization approach:

- **Target**: Force ViT model to classify any image as "spaghetti" (ImageNet class 957)
- **Patch Size**: 32×32 pixels
- **Loss Components**:
  - **Classification Loss**: Drives predictions toward target class
  - **Attention Loss**: Encourages model attention on patch region
  - **Perceptual Loss**: Maintains visual realism using LPIPS

**Key Features:**
- Custom ViT wrapper for attention map extraction
- Balanced multi-objective optimization
- 1500 training iterations with Adam optimizer
- Real-time loss monitoring and visualization

### Attack Evaluation (`eval.ipynb`)

- Tests patch effectiveness on 1000 random Food101 test images
- Random patch placement for universal attack evaluation
- Measures Attack Success Rate (ASR)
- Batch processing for efficient evaluation

## 🛡️ Defense System (`defense.py`)

### Multi-Layered Defense Architecture

#### 1. **Input Preprocessing Defense**
- **Gaussian Blur** (σ=1.0): Smooths high-frequency adversarial patterns
- **Random Crop/Resize** (85% crop ratio): Disrupts patch placement and scale
- **Bit Depth Reduction** (6-bit): Quantizes pixel values to remove fine perturbations

#### 2. **Ensemble Defense**
- **Multiple Architectures**: ViT-Base + ResNet50
- **Soft Voting**: Averages probability distributions
- **Robustness**: Requires fooling multiple different models simultaneously

#### 3. **Integrated Evaluation System**
- Comprehensive metrics calculation
- Visual comparison generation
- Defense effectiveness measurement

### Defense Metrics

- **Clean Accuracy**: Performance on unmodified images
- **Adversarial Accuracy**: Performance on patch-attacked images
- **Defense Improvement**: Percentage point improvement with defense
- **Attack Success Rate**: Fraction of successful attacks
- **Defense Effectiveness**: Percentage reduction in attack success

## 📊 Expected Results

### Attack Performance
- **Training**: Converges to balanced loss (~16.4 total loss)
- **Evaluation**: Variable success rate depending on patch quality
- **Visualization**: Clear patch visibility in attacked images

### Defense Performance
- **Preprocessing Defense**: 20-40% improvement in adversarial accuracy
- **Ensemble Defense**: Additional 10-20% robustness gain
- **Combined System**: 50-80% reduction in attack success rate
- **Clean Performance**: Maintains >90% clean accuracy

## 🔧 Technical Details

### Model Architecture
- **Primary Model**: Vision Transformer (vit_base_patch16_224)
- **Input Size**: 224×224×3
- **Patch Size**: 16×16 (ViT patches)
- **Adversarial Patch**: 32×32×3

### Dataset
- **Training/Evaluation**: Food101 dataset
- **Classes**: 101 food categories
- **Split**: Test split (25,250 images)
- **Preprocessing**: Standard ImageNet normalization

### Optimization
- **Optimizer**: Adam (lr=0.001)
- **Iterations**: 1500 for patch training
- **Batch Size**: 16
- **Device**: CUDA-enabled GPU recommended

## 📈 Usage Examples

### Basic Attack Generation
```python
# Load model and create custom ViT wrapper
model = CustomViT('vit_base_patch16_224')

# Initialize adversarial patch
adversarial_patch = torch.rand(3, 32, 32, requires_grad=True)

# Optimize patch with multi-objective loss
for iteration in range(1500):
    loss = classification_loss + attention_loss + perceptual_loss
    loss.backward()
    optimizer.step()
```

### Defense Evaluation
```python
# Initialize defense system
defense_system = IntegratedDefenseSystem()
defense_system.setup_defenses()

# Apply defense to adversarial images
defended_images = [defense_system.defend_image(img) for img in adversarial_images]

# Evaluate effectiveness
results = defense_system.evaluate_defense(clean_images, adversarial_images, labels, model)
```

## 🎓 Educational Value

This project demonstrates:

1. **Adversarial ML Concepts**: Practical implementation of adversarial attacks
2. **Defense Strategies**: Multiple complementary defense mechanisms
3. **Model Robustness**: Evaluation of ML model vulnerabilities
4. **Computer Vision**: Advanced techniques in image classification
5. **Research Methods**: Systematic attack/defense evaluation

## ⚠️ Ethical Considerations

This project is for **educational and research purposes only**. The techniques demonstrated should be used responsibly:

- Understanding ML model vulnerabilities
- Developing robust AI systems
- Academic research in adversarial ML
- Security testing of deployed models

**Do not use for malicious purposes or unauthorized attacks on production systems.**

## 🔬 Research Applications

- **Adversarial Robustness**: Benchmark defense mechanisms
- **Model Security**: Evaluate production model vulnerabilities  
- **Defense Development**: Test new defense strategies
- **Academic Research**: Reproducible adversarial ML experiments

## 🤝 Contributing

Feel free to extend this project with:
- Additional defense mechanisms
- Different attack strategies
- New model architectures
- Improved evaluation metrics

## 📚 References

- Vision Transformer: "An Image is Worth 16x16 Words" (Dosovitskiy et al.)
- Adversarial Patches: "Adversarial Patch" (Brown et al.)
- LPIPS Perceptual Loss: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (Zhang et al.)
- Food101 Dataset: "Food-101 – Mining Discriminative Components with Random Forests" (Bossard et al.)

## 📄 License

This project is for educational use. Please respect the licenses of the underlying models and datasets used.

---

**Note**: This implementation is designed for learning and research. Results may vary based on hardware, random initialization, and hyperparameter settings.