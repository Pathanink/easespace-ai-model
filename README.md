# EaseSpace AI Model

<div align="center">

**[ภาษาไทย](#-ภาษาไทย) | [English](#-english)**

</div>

---

## 🇹🇭 ภาษาไทย

### สารบัญ
- [เกี่ยวกับโปรเจกต์](#เกี่ยวกับโปรเจกต์)
- [รายละเอียดชุดข้อมูล](#รายละเอียดชุดข้อมูล)
- [สิทธิ์การใช้งาน](#สิทธิ์การใช้งาน)
- [สิ่งที่ต้องเตรียมก่อนเริ่มต้น](#สิ่งที่ต้องเตรียมก่อนเริ่มต้น)
- [การติดตั้งและรันโมเดล](#การติดตั้งและรันโมเดล)
  - [1. ดาวน์โหลดซอร์สโค้ด](#1-ดาวน์โหลดซอร์สโค้ดโปรเจกต์)
  - [2. สร้าง Virtual Environment](#2-สร้างและเปิดใช้งาน-virtual-environment)
  - [3. ติดตั้งแพ็กเกจ](#3-ติดตั้งแพ็กเกจและไลบรารีที่จำเป็น)
  - [4. รันสคริปต์ฝึกสอนโมเดล](#4-รันสคริปต์ฝึกสอนโมเดล)
- [ผลลัพธ์และการนำไปใช้งาน](#ผลลัพธ์การฝึกสอนและการนำไปใช้งาน)

---

### เกี่ยวกับโปรเจกต์

ซอร์สโค้ดสำหรับกระบวนการเทรนและปรับแต่งความแม่นยำ (Fine-Tuning) ของโมเดลปัญญาประดิษฐ์ (WangchanBERTa) สำหรับทำนายอารมณ์จากข้อความ

---

### รายละเอียดชุดข้อมูล

โปรเจกต์นี้ใช้ชุดข้อมูลข้อความภาษาไทยที่รวบรวมและจัดทำขึ้นเพื่อการวิเคราะห์อารมณ์ (Sentiment Analysis) จำนวนทั้งหมด **3,821 ข้อความ**

**ตัวอย่างข้อมูล**

```text
วันนี้โคตรเฟล ทำอะไรก็ผิดไปหมด | neg
ภูมิใจที่เห็นธุรกิจเล็กๆ ที่เริ่มจากศูนย์ค่อยๆ เติบโตขึ้นด้วยน้ำพักน้ำแรงของตัวเอง | pos
```

หมวดหมู่อารมณ์ (Labels) แบ่งออกเป็น 3 ระดับ ได้แก่

| Label | ความหมาย |
|-------|-----------|
| **neg** | อารมณ์เชิงลบ / ความเครียด / ความกังวล |
| **neu** | อารมณ์เป็นกลาง / ข้อความทั่วไป |
| **pos** | อารมณ์เชิงบวก / ความสุข / ความผ่อนคลาย |

---

### สิทธิ์การใช้งาน

ชุดข้อมูล `dataset.txt` ที่แนบมาในโปรเจกต์นี้ **ไม่มีลิขสิทธิ์** นักพัฒนาและผู้ที่สนใจสามารถนำชุดข้อมูลนี้ไปใช้งาน ศึกษา ดัดแปลง หรือเผยแพร่ต่อยอดในโปรเจกต์อื่นๆ ได้อย่างอิสระโดยไม่ต้องขออนุญาต

---

### สิ่งที่ต้องเตรียมก่อนเริ่มต้น

1. โปรแกรม **Visual Studio Code**
2. **Python 3.10** (บังคับใช้เวอร์ชันนี้ เพื่อป้องกันปัญหาไลบรารีขัดแย้งกัน)
3. **Git** (สำหรับ clone โปรเจกต์)

---

### การติดตั้งและรันโมเดล

#### 1. ดาวน์โหลดซอร์สโค้ดโปรเจกต์

เปิด Terminal หรือ Command Prompt แล้วรันคำสั่ง

```cmd
git clone https://github.com/Pathanink/easespace-ai-model.git
```

จากนั้นเข้าสู่โฟลเดอร์โปรเจกต์

```cmd
cd easespace-ai-model
```

#### 2. สร้างและเปิดใช้งาน Virtual Environment

เปิดหน้าต่าง Terminal ใน Visual Studio Code (ตั้งค่าประเภทเป็น Command Prompt) และรันคำสั่งต่อไปนี้

```cmd
py -3.10 -m venv venv
```

```cmd
venv\Scripts\activate
```

> เมื่อเปิดใช้งานสำเร็จ จะเห็นคำว่า `(venv)` ขึ้นนำหน้าบรรทัดคำสั่ง

#### 3. ติดตั้งแพ็กเกจและไลบรารีที่จำเป็น

```cmd
pip install -r requirements.txt
```

#### 4. รันสคริปต์ฝึกสอนโมเดล

ตรวจสอบความเรียบร้อยของไฟล์ชุดข้อมูล จากนั้นเริ่มต้นกระบวนการเทรนด้วยคำสั่ง

```cmd
py -3.10 train_model.py
```

---

### ผลลัพธ์การฝึกสอนและการนำไปใช้งาน

เมื่อการฝึกสอนเสร็จสมบูรณ์ ระบบจะแสดงสรุปผลลัพธ์ความแม่นยำบนหน้าจอ Terminal และสร้างโฟลเดอร์ผลลัพธ์ขึ้นมาอัตโนมัติ (เช่น `wangchanberta_[วันที่และเวลา]`) ซึ่งภายในประกอบด้วย

- **ไฟล์ประเมินผล** — กราฟสถิติ `.png` (Learning Curves, Confusion Matrix) และไฟล์สรุป `.json`
- **โฟลเดอร์ `final_model/`** — ไฟล์น้ำหนักโมเดลและไฟล์การตั้งค่าที่พร้อมใช้งาน

นักพัฒนาสามารถนำโฟลเดอร์ `final_model` ไปวางไว้ในโปรเจกต์ [easespace-webapp](https://github.com/Pathanink/easespace-webapp) เพื่อใช้ประมวลผลการวิเคราะห์อารมณ์บนเว็บแอปพลิเคชันต่อไป

---

<br>

## 🇬🇧 English

### Table of Contents
- [About](#about)
- [Dataset Details](#dataset-details)
- [License](#license)
- [Prerequisites](#prerequisites)
- [Setup and Training](#setup-and-training)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create Virtual Environment](#2-create-and-activate-virtual-environment)
  - [3. Install Dependencies](#3-install-required-packages-and-libraries)
  - [4. Run the Training Script](#4-run-the-training-script)
- [Output and Deployment](#training-output-and-deployment)

---

### About

Source code for the training and Fine-Tuning process of the WangchanBERTa AI model for emotion prediction from text.

---

### Dataset Details

This project uses a Thai-language text dataset compiled specifically for Sentiment Analysis, containing a total of **3,821 entries**.

**Sample Data**

```text
วันนี้โคตรเฟล ทำอะไรก็ผิดไปหมด | neg
ภูมิใจที่เห็นธุรกิจเล็กๆ ที่เริ่มจากศูนย์ค่อยๆ เติบโตขึ้นด้วยน้ำพักน้ำแรงของตัวเอง | pos
```

Emotion labels are divided into 3 categories:

| Label | Description |
|-------|-------------|
| **neg** | Negative emotion / Stress / Anxiety |
| **neu** | Neutral emotion / General text |
| **pos** | Positive emotion / Happiness / Relaxation |

---

### License

The `dataset.txt` file included in this project is **license-free**. Developers and interested parties are free to use, study, modify, or redistribute this dataset in other projects without requiring permission.

---

### Prerequisites

1. **Visual Studio Code**
2. **Python 3.10** (this exact version is required to prevent library conflicts)
3. **Git** (for cloning the project)

---

### Setup and Training

#### 1. Clone the Repository

Open a Terminal or Command Prompt and run:

```cmd
git clone https://github.com/Pathanink/easespace-ai-model.git
```

Then navigate into the project folder:

```cmd
cd easespace-ai-model
```

#### 2. Create and Activate Virtual Environment

Open a Terminal in Visual Studio Code (set the terminal type to **Command Prompt**) and run the following commands:

```cmd
py -3.10 -m venv venv
```

```cmd
venv\Scripts\activate
```

> Once activated successfully, you will see `(venv)` at the beginning of the command line.

#### 3. Install Required Packages and Libraries

```cmd
pip install -r requirements.txt
```

#### 4. Run the Training Script

Verify that the dataset file is in place, then start the training process:

```cmd
py -3.10 train_model.py
```

---

### Training Output and Deployment

Once training is complete, the terminal will display an accuracy summary and automatically generate a new output folder (e.g. `wangchanberta_[date-time]`) containing:

- **Evaluation files** — Statistical charts in `.png` format (Learning Curves, Confusion Matrix) and a `.json` summary file
- **`final_model/` folder** — Model weight files and configuration files ready for use

The `final_model` folder can be placed directly into the [easespace-webapp](https://github.com/Pathanink/easespace-webapp) project to enable emotion analysis processing in the web application.
