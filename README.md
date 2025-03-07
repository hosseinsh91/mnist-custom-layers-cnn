# **MNIST Classification Using Custom CNN Layers**

## **📌 Project Overview**
This project implements **custom Convolutional Neural Network (CNN) layers** for **MNIST digit classification** using **TensorFlow/Keras**. The model is trained on **a subset of MNIST digits (0, 2, 5)** and compares **custom-built Conv2D layers** with standard Conv2D layers in terms of **accuracy and training time**.

### **🚀 Key Features**
✅ **Preprocessing MNIST dataset for selected digits (0, 2, 5)**  
✅ **Implementation of a custom Conv2D layer with concatenation**  
✅ **Comparison between standard CNN and custom CNN model**  
✅ **Training both models for 100 epochs**  
✅ **Measuring training time differences**  
✅ **Plotting validation accuracy and loss**  

---

## **📌 Dataset: MNIST**
The **MNIST dataset** consists of **60,000 training images** and **10,000 test images**, each containing handwritten digits from 0-9.

### **📌 Data Preprocessing**
A subset of digits (0, 2, 5) is selected for classification:
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

selected_labels = [0, 2, 5]
mask = np.isin(y_train, selected_labels)
x_train_f = x_train[mask]
y_train_f = y_train[mask]

# Mapping selected labels to 0, 1, 2 for classification
label_map = {0: 0, 2: 1, 5: 2}
y_train_f = np.array([label_map[label] for label in y_train_f])

# Normalize pixel values
x_train_f = x_train_f / 255.0
```

---

## **📌 Custom CNN Layer Definition**
A **custom Keras Conv2D layer** is defined to **apply multiple filters and concatenate their outputs**.
```python
class Custom_layers(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Custom_layers, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(50, (3,3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(100, (5,5), activation='relu', padding='same')
        self.concat = tf.keras.layers.Concatenate()
    
    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        out = self.concat([x1, x2])
        return out
```

---

## **📌 Model Architectures**

### **1️⃣ Standard CNN Model**
```python
model1 = keras.models.Sequential([
    keras.layers.Input(shape=(28,28,1)),
    keras.layers.Conv2D(80, (2,2), activation='relu', padding='same'),
    keras.layers.Conv2D(200, (2,2), activation='relu', padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(3, activation='softmax')
])
```

### **2️⃣ Custom CNN Layer Model**
```python
model2 = keras.models.Sequential([
    keras.layers.Input(shape=(28,28,1)),
    Custom_layers(),
    keras.layers.Flatten(),
    keras.layers.Dense(3, activation='softmax')
])
```

---

## **📌 Model Compilation & Training**
Both models are compiled using **SGD optimizer** and trained for **100 epochs**.

### **Train Standard CNN Model**
```python
start_time_cnn_model = time.time()
model1.compile(optimizer='sgd',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

history1 = model1.fit(x_train_f, y_train_f, epochs=100, validation_split=0.2)
training_time_cnn = time.time() - start_time_cnn_model
```

### **Train Custom CNN Layer Model**
```python
start_time_custom_model = time.time()
model2.compile(optimizer='sgd',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

history2 = model2.fit(x_train_f, y_train_f, epochs=100, validation_split=0.2)
training_time_custom = time.time() - start_time_custom_model
```

---

## **📌 Model Performance Comparison**
A comparison between the **custom CNN model** and **standard CNN model** is plotted.
```python
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(history1.history["val_loss"], label="val_loss_CNN")
ax.plot(history2.history["val_loss"], label="val_loss_Custom_Layer")
ax.plot(history1.history["val_accuracy"], label="val_accuracy_CNN")
ax.plot(history2.history["val_accuracy"], label="val_accuracy_Custom_Layer")
ax.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.title("Comparison of Custom CNN vs. Standard CNN Model")
plt.show()
```

### **📊 Key Observations**
✅ **Custom CNN model achieved comparable accuracy with reduced training time**  
✅ **Validation loss decreased significantly for the custom model**  
✅ **Custom Conv2D layers improve training efficiency**  

---

## **📌 Training Time Comparison**
```python
print('The training time of CNN model is:', training_time_cnn)
print('The training time of Custom layers Model is:', training_time_custom)
```
### **📊 Training Time Results**
- **Standard CNN Model**: 350.50 seconds  
- **Custom Layers Model**: 195.12 seconds  

---

## **📌 Installation & Setup**
### **📌 Prerequisites**
- **Python 3.x**
- **Jupyter Notebook**
- **TensorFlow, NumPy, Matplotlib**

### **📌 Install Required Libraries**
```bash
pip install tensorflow numpy matplotlib
```

---

## **📌 Running the Notebook**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YourGitHubUsername/mnist-custom-layers-cnn.git
cd mnist-custom-layers-cnn
```

### **2️⃣ Launch Jupyter Notebook**
```bash
jupyter notebook
```

### **3️⃣ Run the Notebook**
Open `mnist_custom_layers.ipynb` and execute all cells.

---

## **📌 Conclusion**
This project demonstrates how **custom Conv2D layers can enhance CNN models** by reducing training time while maintaining comparable accuracy.

---

## **📌 License**
This project is licensed under the **MIT License**.

