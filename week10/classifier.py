
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from skimage import color
from skimage.feature import hog


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
data=unpickle('cifar-10-batches-py/data_batch_1')
print(data.keys())
images_list = data[b'data']
labels_list = data[b'labels']

# Visualize the 4th image
img = images_list[3]
img = np.array(img).reshape(3, 32, 32).transpose(1, 2, 0)  # Convert to (32, 32, 3)

plt.imshow(img)
plt.title(f"Image: {labels_list[3]}")
plt.axis('off')
plt.show()

# train test split
x_train, x_test, y_train, y_test = train_test_split(images_list, labels_list, test_size=0.2, random_state=42)

# way1
# piple = Pipeline([
#     ('scaler', StandardScaler()),
#     ('pca', PCA(1024)),
#     ('model', RandomForestClassifier(verbose=True, n_jobs=-1, n_estimators=100))
# ])

# piple.fit(x_train, y_train)
# y_predict = piple.predict(x_test)
# #classification report
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_predict))

# use hog features
pipeline = Pipeline([
    ('model', RandomForestClassifier(verbose=True, n_jobs=-1, n_estimators=100))
])
def extract_hog_features(images):
    hog_features = []
    for img in images:
        img = np.array(img).reshape(3, 32, 32).transpose(1, 2, 0)  # Convert to (32, 32, 3)
        img_gray = color.rgb2gray(img)  # Convert to grayscale
        features = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)
x_train_hog = extract_hog_features(x_train)
x_test_hog = extract_hog_features(x_test)
pipeline.fit(x_train_hog, y_train)
y_predict = pipeline.predict(x_test_hog)
# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
