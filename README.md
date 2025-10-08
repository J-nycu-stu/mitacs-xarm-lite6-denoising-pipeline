# Mitacs xArm Lite 6 Motion Control & Denoising Pipeline  
**Python-based robotic motion control and video denoising framework**

---

## Overview

This repository contains two related subprojects developed for the **xArm Lite 6** robotic arm during a Mitacs research initiative:

1. **Robotic Motion Control** – Python scripts for parametric trajectory control of the xArm Lite 6.  
2. **Video Denoising Pipeline** – Jupyter notebook for processing 24 fps footage of robotic experiments using OpenCV.

---

## 1️⃣ Robotic Motion Control

This part of the project focuses on **precise actuation and motion control** of the xArm Lite 6 through programmable trajectories.

### Features
- Supports **linear** and **arc** motion paths defined by mathematical parameters.  
- Integrates **press-in / dwell / press-out** sequences for consistent motion execution.  
- Implements **approach shim** and **tqdm-based tracking** for better repeatability and feedback.  

### Files
- **`line_motion_press_final.py`** – Executes linear motion trajectories for the robot arm.  
- **`arc_motion_press_final.py`** – Executes arc motion trajectories for curved path movements.  

### Demo Videos
*(Note: Large files — please download the raw files to view them.)*  
1. **`Arc_motion.mp4`** – Demonstration of arc motion on the physical xArm Lite 6.  
2. **`Line_motion.mp4`** – Demonstration of linear motion on the physical xArm Lite 6.  

---

## 2️⃣ Video Denoising Pipeline

This part of the project develops a **reproducible OpenCV-based denoising workflow** for videos recorded from robotic experiments.

### Features
- Applies **color NL-Means filtering**, **CLAHE (adaptive histogram equalization)**, and **mild gamma correction**.  
- Reassembles denoised frames into **24 fps video** for analysis and comparison.  
- Enables **visual inspection** of noise reduction performance.  

### Files
- **`image_process.ipynb`** – Jupyter notebook for implementing and visualizing the denoising process.  

### Demo Videos
*(Note: Large files — please download the raw files to view them.)*  
**`original vs. processed.mp4`** – Side-by-side comparison of unprocessed(left side) and denoised(right side) videos.  
