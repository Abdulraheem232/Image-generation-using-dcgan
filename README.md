
---

````markdown
# 🧠 Image Generation using DCGAN on CIFAR-10

This project demonstrates how to generate synthetic images using a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on the **CIFAR-10** dataset. It is implemented entirely in **Jupyter Notebook** using **PyTorch**.

---

## 📓 Notebook

- `Dcgan_and_image_generation.ipynb`: The main notebook containing all code for loading data, building the DCGAN model, training, and visualizing generated images.

---

## 📚 Dataset

- **Name**: CIFAR-10  
- **Source**: `torchvision.datasets.CIFAR10`  
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)  
- **Image Size**: 32x32 RGB  

The dataset is automatically downloaded using PyTorch's `torchvision`.

---

## 🧠 DCGAN Architecture

### 🎨 Generator
- Input: Random noise vector (latent vector `z`, usually 100-dimensional)
- Upsamples using transposed convolution layers
- Uses BatchNorm + ReLU
- Final layer uses Tanh to output a 3×32×32 image

### 🛡️ Discriminator
- Input: 3×64×64 image
- Downsamples using regular convolutions
- Uses BatchNorm + LeakyReLU
- Final layer outputs a single value with Sigmoid (real or fake)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Image-generation-using-dcgan.git
cd Image-generation-using-dcgan
````

### 2. Install Dependencies

```bash
pip install torch torchvision matplotlib notebook
```

### 3. Run the Notebook

```bash
jupyter notebook Dcgan_and_image_generation.ipynb
```

Follow the cells in order to:

* Load the dataset
* Define and initialize the models
* Train the GAN
* Generate and view sample images

---


---

## 📁 Folder Structure

```
Image-generation-using-dcgan/
├── Dcgan_and_image_generation.ipynb
├── samples/                 # Generated images from training
├── README.md
└── requirements.txt         # Optional
```

---

## ⚙️ Training Config

* Epochs: 50
* Batch Size: 128
* Learning Rate: 0.0002
* Optimizer: Adam (`betas=(0.5, 0.999)`)
* Latent Dimension (z): 100

---

## 📚 References

* [DCGAN Original Paper](https://arxiv.org/abs/1511.06434)
* [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
* [CIFAR-10 Dataset Info](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 👤 Author

**Your Name**
GitHub: [@your-username](https://github.com/your-username)

---

## 📄 License

This project is licensed under the **MIT License**.

```

---

```
