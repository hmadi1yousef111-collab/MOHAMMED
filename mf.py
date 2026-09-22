import cv2
import numpy as np
import os

# 1. قراءة الصورة
image_path = r"C:\Users\PC-LAB1\Desktop\immge processing\20260922_123631.png"
image = cv2.imread(image_path)

if image is None:
    print("خطأ: تعذر قراءة الصورة! تأكد من صحة المسار.")
else:
    # 2. تحويل الصورة للتدرج الرمادي والتنعيم
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- تطبيق كافة أنواع العتبات (Thresholding Types) ---

    # 1. العتبة الثابتة العادية (Binary) وعكسها (Binary Inverted)
    _, t_binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    t_binary_inv = cv2.bitwise_not(t_binary)

    # 2. العتبة المقطوعة (Truncate) وعكسها
    _, t_trunc = cv2.threshold(blurred, 127, 255, cv2.THRESH_TRUNC)
    t_trunc_inv = cv2.bitwise_not(t_trunc)

    # 3. العتبة الصفرية (To Zero) وعكسها
    _, t_tozero = cv2.threshold(blurred, 127, 255, cv2.THRESH_TOZERO)
    t_tozero_inv = cv2.bitwise_not(t_tozero)

    # 4. عتبة أوتسو (Otsu) وعكسها
    _, t_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_otsu_inv = cv2.bitwise_not(t_otsu)

    # 5. عتبة المثلث (Triangle) وعكسها
    _, t_triangle = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    t_triangle_inv = cv2.bitwise_not(t_triangle)

    # 6. العتبة التكيفية الجاوصية (Adaptive Gaussian) وعكسها
    t_adapt_gauss = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    t_adapt_gauss_inv = cv2.bitwise_not(t_adapt_gauss)

    # --- تجهيز قائمة الصور مع العناوين ---
    threshold_list = [
        ("Original Gray", gray),
        ("1. Binary", t_binary),
        ("1. Binary Inverted", t_binary_inv),
        ("2. Truncate", t_trunc),
        ("2. Truncate Inverted", t_trunc_inv),
        ("3. To Zero", t_tozero),
        ("3. To Zero Inverted", t_tozero_inv),
        ("4. Otsu Auto", t_otsu),
        ("4. Otsu Inverted", t_otsu_inv),
        ("5. Triangle Auto", t_triangle),
        ("5. Triangle Inverted", t_triangle_inv),
        ("6. Adaptive Gauss", t_adapt_gauss),
        ("6. Adaptive Inverted", t_adapt_gauss_inv)
    ]

    labeled_images = []

    for title, img in threshold_list:
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # إضفاء شريط أبيض خلفي للكتابة بوضوح
        cv2.rectangle(color_img, (0, 0), (color_img.shape[1], 35), (255, 255, 255), -1)
        cv2.putText(color_img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        labeled_images.append(color_img)

    # إنشاء مربع فارغ للموازنة في الشبكة (3x4 = 12 صورة)
    h, w = gray.shape
    blank_tile = np.zeros((h, w, 3), dtype=np.uint8)

    # --- تجميع الصور في شبكة 3x4 (3 صفوف × 4 أعمدة) ---
    row1 = np.hstack(labeled_images[0:4])
    row2 = np.hstack(labeled_images[4:8])
    row3 = np.hstack(labeled_images[8:12])

    combined_grid = np.vstack([row1, row2, row3])

    # --- حفظ النتيجة وعرضها في نافذة واحدة ---
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/all_thresholds_with_inversion.jpg", combined_grid)

    cv2.namedWindow("All Thresholding Types (Original vs Inverted)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("All Thresholding Types (Original vs Inverted)", 1280, 800)
    cv2.imshow("All Thresholding Types (Original vs Inverted)", combined_grid)

    print("تم تجميع جميع الصور (العادية والمعكوسة) في صورة واحدة بنجاح!")
    print("اضغط على أي زر لإغلاق النافذة...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()