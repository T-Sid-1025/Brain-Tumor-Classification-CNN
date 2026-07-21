# Brain Tumor Classification using CNN

## What is This Project?

This is a deep learning project that uses a neural network to analyze brain MRI scans and detect if there's a tumor or not. Think of it as an AI assistant for doctors - it looks at MRI images and tells you what it thinks is in the image.

The project has two parts:
1. **A machine learning model** - The AI that learns to recognize tumors
2. **A web application** - Where you can upload an MRI image and get results

---

## How Well Does It Work?

The model was trained on 3,064 brain MRI images and achieved:
- **76.74% accuracy** on test data
- Can detect tumors with 86% success rate
- Takes less than 1 second to analyze an image

**What this means:** Out of 100 MRI scans, the AI correctly identifies about 77 of them. It's pretty good but not perfect - so doctors still need to double-check.

---

## What's Inside?

### The Model
- Uses a 4-layer convolutional neural network (CNN)
- Input: Brain MRI images (150×150 pixels)
- Output: Classification as Normal, Tumor, or Unclear
- Built with TensorFlow and Keras

### The Web App
- Made with Streamlit (makes it easy to create web apps)
- Has 4 tabs:
  - **Dashboard** - See system info and accuracy
  - **Analysis** - Upload an MRI and get AI prediction
  - **Performance** - View training results and graphs
  - **Guidelines** - Read important info about the system

---

## How to Use This

### Installation (First Time Setup)

1. **Download the code**
```bash
git clone https://github.com/T-Sid-1025/Brain-Tumor-Classification-CNN.git
cd Brain-Tumor-Classification-CNN
```

2. **Install required software**
```bash
pip install tensorflow streamlit pillow matplotlib pandas numpy scikit-learn jupyter
```

3. **Get the trained model** - Two options:

   **Option A: Train it yourself (takes ~60 minutes)**
   ```bash
   jupyter notebook brain_tumor_final.ipynb
   ```
   Then run all the cells in the notebook

   **Option B: Ask for the pre-trained model**
   - Contact the developer and they'll give you the trained model file
   - Place it in the project folder

4. **Run the web app**
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

---

## How to Use the App

### Dashboard Tab
Just shows you information about the model - like how accurate it is, how much data it was trained on, etc.

### Analysis Tab
1. Enter patient ID (optional)
2. Enter scan date
3. Enter radiologist name (optional)
4. Upload an MRI image (jpg or png)
5. The AI analyzes it and shows:
   - What it thinks (Normal/Tumor/Unclear)
   - Confidence level (0-100%)
   - A bar chart showing confidence for all 3 options

### Performance Tab
Shows graphs of how the model performed:
- Confusion matrix (which predictions were right/wrong)
- Training curves (how the model improved over time)
- Performance stats for each category

### Guidelines Tab
Read important information about how to use the system and what it can/can't do.

---

## The Training Process

The model was trained like this:

1. **Loaded 3,064 MRI images** - 46.5% normal brains, 23.1% with tumors, 30.4% unclear
2. **Split the data** - 70% for training, 15% for validation, 15% for testing
3. **Trained for 50 epochs** - But stopped at epoch 29 because the accuracy stopped improving
4. **Final results** - 76.74% accuracy on test data

---

## Important Things to Know

### This is NOT a Doctor
- This system is for learning, not for actual medical diagnosis
- A real doctor needs to look at results and make the final call
- The AI can miss some tumors (85% of them it catches, 15% it misses)
- Don't use this to diagnose real patients without a radiologist

### Limitations
- Works best with clear MRI images
- Trained on only 3,064 images - might not work with unusual cases
- 76.74% accuracy means it's wrong sometimes
- Some normal brains get flagged as tumors (false alarms)

---

## What You'll See in the Folder

```
Project Folder/
├── brain_tumor_final.ipynb    → Jupyter notebook with all the code
├── app.py                     → The web app
├── .gitignore                 → Files to ignore (tells git not to upload big files)
├── README.md                  → This file
└── Images/
    ├── training_history.png    → Graphs showing how training went
    ├── confusion_matrix.png    → Shows what the model got right/wrong
    ├── class_distribution.png  → Shows split of normal vs tumor vs unclear
    └── Other graphs...
```

Model files (too big for GitHub):
- `best_brain_tumor_model.h5` - The trained model (128 MB)
- You need this to run the app

---

## The Technology

**Python 3.10** - Programming language  
**TensorFlow** - AI framework that handles neural networks  
**Keras** - Simpler interface for building neural networks  
**Streamlit** - Framework for making web apps without HTML/CSS  
**NumPy & Pandas** - Libraries for handling data  
**Matplotlib** - For making graphs and visualizations  

---

## How the AI Sees Images

The model looks at MRI images like this:

1. First layer (32 filters) - Detects simple patterns like edges and corners
2. Second layer (64 filters) - Detects shapes and combinations of patterns
3. Third layer (128 filters) - Detects larger structures
4. Fourth layer (256 filters) - Detects full features like "tumor-like area"
5. Then it makes a decision - Normal? Tumor? Unclear?

It's like how humans learn - start with simple things, combine them into more complex understanding.

---

## Results

**On Normal Brains:** Correctly identified 58 out of 117 (50%)  
**On Tumors:** Correctly identified 180 out of 209 (86%)  
**On Unclear Cases:** Correctly identified 115 out of 134 (88%)  

What this means:
- Great at detecting tumors (won't miss many)
- Sometimes flags normal brains as tumors
- Good at recognizing unclear cases

---

## What The App Actually Shows

When you upload an MRI image:

1. **Green box** - Normal brain (99% confidence)
2. **Red box** - Tumor detected (85% confidence)
3. **Yellow box** - Unclear/needs review (45% confidence)

Then you see bars for each category showing the confidence percentage.

---

## If You Want to Train It Yourself

Open the Jupyter notebook (`brain_tumor_final.ipynb`) and you'll see:

1. Load images from the dataset
2. Show some sample images
3. Check data quality
4. Build the neural network
5. Train for 50 epochs
6. Show results and graphs
7. Save the model

Each section has code and explanations. You can run them one by one.

---

## Quick Troubleshooting

**"Module not found error"** - Run `pip install tensorflow streamlit` etc.

**"Model file not found"** - You need to train it first or get the file from the developer

**"App won't open"** - Make sure you're in the right folder and ran `streamlit run app.py`

**"Image upload not working"** - Try a JPG or PNG file

---

## Contact

If you have questions or want the pre-trained model:
- GitHub: https://github.com/T-Sid-1025
- Email: tagare.siddhant95@gmail.com

---

## One Last Important Thing

**This project is for learning and research only.** Don't use it to diagnose real patients. If you want to do that, you need doctors' approval, more training, and proper medical certification.

The AI is helpful but not perfect. Always use it alongside actual medical professionals.

---

**That's it! This is a brain tumor detection AI. It's cool, it works pretty well, but remember - it's an assistant to doctors, not a replacement for them.**
