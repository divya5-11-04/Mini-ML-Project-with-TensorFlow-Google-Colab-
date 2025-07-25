# 🧠 TensorFlow Qwik Start: My First Machine Learning Model

This project demonstrates a simple linear regression model using **TensorFlow** and **Google Colab** to learn the relationship between input and output values. It's a practical implementation of the classic "Hello World" in machine learning using a single-layer neural network.

---

## 🚀 Project Overview

The objective of this project was to build, train, and test a machine learning model that learns a linear relationship between `x` and `y` values:

X = [-1, 0, 1, 2, 3, 4]
Y = [-2, 1, 4, 7, 10, 13]


As seen, the relationship is:  
**Y = 3X + 1**

---

## 📦 Tech Stack

- **Language**: Python 3  
- **Framework**: TensorFlow (Keras API)  
- **Libraries**: NumPy  
- **Environment**: Jupyter Notebook (Google Colab)

---

## 📚 Concepts Learned

- Supervised Learning (Linear Regression)
- Neural Network (Single Dense Layer)
- Loss Functions (`MeanSquaredError`)
- Optimizers (`Stochastic Gradient Descent`)
- Model Training (`model.fit`)
- Prediction & Inference (`model.predict`)
- Understanding loss reduction over epochs

---

## 🧪 How the Model Works

1. **Data Preparation**
```python
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
Model Design

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
Model Compilation

model.compile(optimizer='sgd', loss='mean_squared_error')
Training

model.fit(xs, ys, epochs=500)
Prediction

print(model.predict([10.0]))
📊 Output
After training, the model predicts:

[[30.999973]]
Which is very close to the expected output 31 for input 10.

🛠 Skills Demonstrated
TensorFlow Model Building

Neural Network Training & Evaluation

Python + NumPy Data Handling

Jupyter Notebook Workflow

Machine Learning Concept Application

📁 File Structure

.
├── model.ipynb           # Jupyter Notebook with all code
├── README.md             # This file
🧠 Future Work
Expand to non-linear datasets

Implement multi-layer neural networks

Try classification problems (e.g., MNIST or Fashion MNIST)

Experiment with different optimizers and loss functions

🙋‍♀️ Author
Divya Monga
Third-Year Engineering Student | Robotics & AI
Skilled in Python, TensorFlow, Full-Stack Development, and Applied AI

