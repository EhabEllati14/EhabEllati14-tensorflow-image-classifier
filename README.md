# 🌸 Image Classifier using TensorFlow

## 📌 Project Overview
This project is an **Image Classifier** built using **TensorFlow** and **TensorFlow Hub**, trained on the **Oxford Flowers 102 dataset**. The model leverages **MobileNetV2** for feature extraction and classifies flower species with high accuracy.

## 🚀 Features
- Pretrained **MobileNetV2** feature extractor
- Image preprocessing & augmentation
- Command-line prediction script
- Top-K predictions support
- Class label mapping with `label_map.json`

## 📂 Project Structure
```
image-classifier/
│── model/                   # Saved trained model
│── data/                    # Dataset (if applicable)
│── predict.py               # Script for making predictions
│── train.py                 # Script for training the model
│── label_map.json           # Mapping of class indices to labels
│── requirements.txt         # Python dependencies
│── README.md                # Project documentation
```

## 🛠 Installation & Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/image-classifier.git
   cd image-classifier
   ```
2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏋️ Training the Model
Run the following command to train the model:
```bash
python train.py --epochs 10 --learning_rate 0.001 --save_dir model/
```

## 🔍 Making Predictions
To classify an image and get the top-K predictions:
```bash
python predict.py --image_path path/to/image.jpg --model model/flower_classifier.keras --top_k 5 --category_names label_map.json
```

## 📊 Example Output
```
Flower Name: Rose
Probability: 98.76%
...
```

## 🤖 Technologies Used
- **Python 3.12**
- **TensorFlow & TensorFlow Hub**
- **NumPy, Matplotlib, PIL**
- **argparse** for command-line arguments

## 🎯 Future Improvements
- Implementing a **web-based UI** for predictions
- Experimenting with different **pretrained models**
- Deploying the model as a **REST API**

## 🤝 Contributing
Feel free to fork this repository and submit pull requests!

## 📜 License
This project is licensed under the **MIT License**.
