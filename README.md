# 🍕 Generative Modelling Assignment – VS Code Project

This project is a VS Code-ready implementation of GANs across multiple domains, including synthetic data, medical imaging, cybersecurity, and creative AI.

## 📂 Project Structure

gan_vscode_project/
├── .vscode/
├── data/
│   ├── categories.txt
│   ├── pizza.npy
│   └── Wednesday-workingHours.pcap_ISCX.csv
├── notebooks/
├── outputs/
│   ├── bloodmnist/
│   ├── cicids/
│   ├── part1/
│   └── quickdraw/
├── src/
│   ├── __init__.py
│   ├── common.py
│   ├── part1_synthetic_gan.py
│   ├── part2_bloodmnist_dcgan.py
│   ├── part2_cicids_tabular_gan.py
│   └── part2_quickdraw_dcgan.py
├── requirements.txt
├── run_all.py
├── setup.ps1
├── setup.sh
└── README.md

---

## ⚙️ Setup Instructions

### Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

### macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---
Before Running the project add the datasets 
Wednesday-workingHours.pcap_ISCX.csv ====https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
pizza.npy===https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/pizza.npy
(https://github.com/googlecreativelab/quickdraw-dataset)

## 🚀 Running the Project

### Part 1
python src/part1_synthetic_gan.py

### Part 2.1
python src/part2_bloodmnist_dcgan.py

### Part 2.2
python src/part2_cicids_tabular_gan.py --input_csv data/Wednesday-workingHours.pcap_ISCX.csv

### Part 2.3
python src/part2_quickdraw_dcgan.py --input_npy data/pizza.npy

---

## ▶️ Run All
python run_all.py

---

## 📊 Outputs

All outputs are saved in:
outputs/

Includes:
- Generated images
- Loss graphs
- PCA plots
- Metrics

---

## 💡 Notes

- Use fewer epochs for testing
- Increase epochs for final results
- Use generated outputs in report

---

## ⚠️ Troubleshooting

If torch not found:
pip install torch torchvision torchaudio

If file path error:
Check:
data/pizza.npy
data/Wednesday-workingHours.pcap_ISCX.csv

---

## ✅ Summary

This project demonstrates how GANs can learn patterns and generate synthetic data across multiple domains.
