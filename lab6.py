import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def create_dummy_image(text="Image"):
    img = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
    cv2.putText(img, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return img

def apply_rotation(img, angle_range):
    angle = random.uniform(-angle_range, angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_brightness(img, percent_range):
    percent = random.uniform(-percent_range, percent_range)
    factor = 1.0 + percent
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# BÀI 1: Căn hộ / Mặt tiền
def bai1():
    orig_images = [create_dummy_image(f"Can Ho {i+1}") for i in range(5)]
    aug_images = []

    for img in orig_images:
        img_resized = cv2.resize(img, (224, 224))
        
        img_aug = img_resized.copy()
        if random.choice([True, False]):
            img_aug = cv2.flip(img_aug, 1)
        
        img_aug = apply_rotation(img_aug, 15)
        img_aug = apply_brightness(img_aug, 0.20)
        
        img_gray = cv2.cvtColor(img_aug, cv2.COLOR_RGB2GRAY)
        
        img_norm = img_gray.astype(np.float32) / 255.0
        aug_images.append(img_norm)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Bài 1: Ảnh gốc (Trên) - Ảnh Augmentation (Dưới)')
    for i in range(5):
        axes[0, i].imshow(orig_images[i])
        axes[0, i].axis('off')
        axes[1, i].imshow(aug_images[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
    plt.show()

# BÀI 2: Xe ô tô / Xe máy
def bai2():
    img = create_dummy_image("Xe O To")
    
    img_resized = cv2.resize(img, (224, 224))
    
    img_aug = apply_rotation(img_resized, 10)
    
    img_aug = apply_brightness(img_aug, 0.15)
    
    noise = np.random.normal(0, 25, img_aug.shape).astype(np.uint8)
    img_aug = cv2.add(img_aug, noise)
    
    img_gray = cv2.cvtColor(img_aug, cv2.COLOR_RGB2GRAY)
    
    img_norm = img_gray.astype(np.float32) / 255.0
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle('Bài 2: Gốc vs Augmentation (Noise, Rotate, Grayscale)')
    axes[0].imshow(img_resized)
    axes[0].set_title("Resize 224x224")
    axes[0].axis('off')
    axes[1].imshow(img_norm, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Normalized [0-1]")
    axes[1].axis('off')
    plt.show()

# BÀI 3: Trái cây / Nông sản
def bai3():
    img = create_dummy_image("Trai Cay")
    
    aug_grid = []
    for _ in range(9):
        img_aug = cv2.resize(img, (224, 224))
        
        if random.choice([True, False]):
            crop_size = random.randint(150, 200)
            start_x, start_y = random.randint(0, 224-crop_size), random.randint(0, 224-crop_size)
            img_aug = img_aug[start_y:start_y+crop_size, start_x:start_x+crop_size]
            img_aug = cv2.resize(img_aug, (224, 224))
            
        if random.choice([True, False]):
            img_aug = cv2.flip(img_aug, 1)
        img_aug = apply_rotation(img_aug, 30)
        
        img_norm = img_aug.astype(np.float32) / 255.0
        aug_grid.append(img_norm)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle('Bài 3: Grid 3x3 Augmentation')
    for i, ax in enumerate(axes.flat):
        ax.imshow(aug_grid[i])
        ax.axis('off')
    plt.show()

# BÀI 4: Phòng / Nội thất
def bai4():
    img = create_dummy_image("Noi That")
    img_resized = cv2.resize(img, (224, 224))
    
    aug_images = []
    for _ in range(3):
        img_aug = img_resized.copy()
        img_aug = apply_rotation(img_aug, 15)
        if random.choice([True, False]):
            img_aug = cv2.flip(img_aug, 1)
        img_aug = apply_brightness(img_aug, 0.20)
        
        img_gray = cv2.cvtColor(img_aug, cv2.COLOR_RGB2GRAY)
        img_norm = img_gray.astype(np.float32) / 255.0
        aug_images.append(img_norm)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Bài 4: 1 Ảnh Gốc - 3 Ảnh Augmentation')
    axes[0].imshow(img_resized)
    axes[0].set_title("Gốc")
    axes[0].axis('off')
    for i in range(3):
        axes[i+1].imshow(aug_images[i], cmap='gray', vmin=0, vmax=1)
        axes[i+1].set_title(f"Aug {i+1}")
        axes[i+1].axis('off')
    plt.show()

if __name__ == "__main__":
    bai1()
    bai2()
    bai3()
    bai4()
