"""
Introduction:
Các bạn vào trang web sau: https://quickdraw.withgoogle.com/
Các bạn hãy chơi thử game trong vòng 1-2 phút
Đây là game QuickDraw, được phát triển dựa trên bộ dataset nổi tiếng cùng tên của Google. Các bạn có thể vào link sau
https://quickdraw.withgoogle.com/data
để có cái nhìn tổng quát về bộ dataset này. Các bạn có thể click vào 1 icon bất kì để xem các bản vẽ của category tương ứng

Pre-requisites:
Các bạn hãy vào link sau đây
https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false
và tự chọn cho mình 10 categories mà các bạn thích (hint: Đừng chọn category nào quá phức tạp)
Sau đó các bạn hãy download các file tương ứng của các categories này (click vào mũi tên đi xuống ở dòng tương ứng)
Các file được download sẽ có tên như sau: full_numpy_bitmap_{category}.npy. Sau đó các bạn hãy để các file .npy này
vào chung với folder của script python này. Giờ các bạn có thể chuyển sang phần tiếp theo - phần lập trình
"""
import numpy as np
from matplotlib import pyplot as plt

def calculate_dot_product(test_image, avg_image):
    """
    Tính tích vô hướng giữa một bức ảnh test và bức ảnh trung bình.
    :param test_image: Bức ảnh test, kích thước (784,)
    :param avg_image: Bức ảnh trung bình, kích thước (784,)
    :return: Giá trị tích vô hướng
    """
    score = np.dot(test_image, avg_image.flatten())  # Tính tích vô hướng giữa ảnh test và ảnh trung bình
    return score

# Bước 1: Các bạn hãy chọn 1 trong số các file .npy mà các bạn vừa tải về ở trên, và thay đổi đường dẫn tương ứng phía dưới
file_path = "./full_numpy_bitmap_book.npy"  # <= Các bạn thay từ bicycle bằng tên tương ứng của category các bạn chọn nhé
images = np.load(file_path).astype(np.float32)  # Load toàn bộ các ảnh của category này vào biến images
print(images.shape)
train_images = images[:-10]  # Lấy tất cả ảnh, ngoại trừ 10 ảnh cuối ra làm bộ training.
test_images = images[-10:]  # Giữ 10 ảnh cuối làm bộ test

#TODO Bước 2: Các bạn hãy tính ra 1 bức ảnh trung bình của bộ training. Bức ảnh này sẽ có kích thước 28x28 pixel (hint: 784=28*28)
avg_image = train_images.mean(axis=0)  # Tính trung bình của tất cả các ảnh trong train_images
avg_image = avg_image.reshape(28, 28)  # Reshape lại thành kích thước 28x28
# Bước 3: Các bạn sẽ visualize bức ảnh trung bình các bạn vửa tính được ở bước 2 bằng 2 dòng sau. Các bạn thử
# xem các bạn có nhận ra được category mà các bạn chọn bằng cách nhìn vào bức ảnh trung bình này không nhé
# plt.imshow(avg_image)
# plt.show()

# Bước 4: Các bạn chọn 1 index bất kì từ 0 đến 9. Ví dụ mình chọn index = 4
# Sau đó các bạn hãy tính tích vô hướng (dot product) của bức ảnh test này với bức ảnh trung bình các bạn tính được ở trên
index = 5  # Các bạn có thể thay đổi index tùy ý
test_image = test_images[index]
#TODO Các bạn tính tích vô hướng (dot product) của bức ảnh test và ảnh trung bình ở dòng dưới đây
# (các bạn có thể code trên nhiều hơn 1 dòng)
score = np.dot(test_image, avg_image.flatten())  # Tính tích vô hướng giữa ảnh test và ảnh trung bình
#print("Book score %s", score)

# Bước 5: Các bạn hãy lặp lại bước 1 đến 3 cho tất cả các categories còn lại (chú ý tại bước 1 các bạn không cần phân
# ra train với test images nữa nhé, coi như là dùng tất cả cho train). Sau đó các bạn hãy tính tích vô hướng của từng ảnh
# trung bình của ảnh test các bạn chọn ở bước 4 với từng bức ảnh trung bình này.
for i in ["airplane", "apple", "bed", "bucket", "book"]:  # Giả sử có 6 categories khác nhau
    file_path = f"./full_numpy_bitmap_{i}.npy"  # Các bạn thay đổi tên file tương ứng với từng category
    images = np.load(file_path).astype(np.float32)  # Load toàn bộ các ảnh của category này vào biến images
    avg_image = images.mean(axis=0).reshape(28, 28)  # Tính bức ảnh trung bình của category này
    score = np.dot(test_image, avg_image.flatten())  # Tính tích vô hướng với ảnh test đã chọn
    print(f"Score for category {i}: {score}")
# Cuối cùng các bạn xem là liệu trong 10 score này, score tương ứng với tích vô hướng của ảnh test này với
# ảnh trung bình của category của chính nó có phải là score lớn nhất không nhé. Các bức ảnh trung bình mà các bạn tính ra
# có thể xem như là weight cho từng category mà các bạn vừa học ở bài 1 (tất nhiên là weight của mô hình sau khi đã
# train xong)

# Bước 6 (optional): Các bạn thử visualize 10 weight (avg_image) này trong cùng 1 ảnh kích thước 2x5 hoặc 5x2 để so sánh xem,
# weight của các categories nào dễ nhìn và weight nào không nhé
plt.figure(figsize=(10, 4))
plt.suptitle("Average Images of Categories", fontsize=16)
for i, category in enumerate(["airplane", "apple", "bed", "bucket", "book"]):
    file_path = f"./full_numpy_bitmap_{category}.npy"  # Các bạn thay đổi tên file tương ứng với từng category
    images = np.load(file_path).astype(np.float32)  # Load toàn bộ các ảnh của category này vào biến images
    avg_image = images.mean(axis=0).reshape(28, 28)  # Tính bức ảnh trung bình của category này
    score = np.dot(test_image, avg_image.flatten())
    ax = plt.subplot(2, 5, i + 1)
    ax.imshow(avg_image, cmap='gray')
    ax.set_title(category)
    ax.axis('off')

    # Hiển thị score rõ ràng bên dưới ảnh
    ax.text(14, 30, f"Score: {score:.2f}", fontsize=9, ha='center')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
