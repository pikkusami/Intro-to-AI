def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, roc_auc_score, auc
from sklearn.preprocessing import minmax_scale
from skimage.feature import hog
from skimage import io
from skimage.filters import threshold_sauvola
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy import ndimage
from skimage import measure
import matplotlib.patches as mpatches
from imblearn.over_sampling import SVMSMOTE
plt.rcParams['figure.figsize'] = (15, 6)

# Load MNIST data. Data includes 784 variables (28x28 grayscale) and data's class information.
data, classes = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
data, classes = np.asarray(data, 'int16'), np.asarray(classes, 'int')
data, classes = data[:10000], classes[:10000]

# For Images calculate HOG-features
features = []
for sample in data:
    features.append(hog(sample.reshape((28, 28)), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(1, 1), block_norm="L2-Hys", visualize=False))

# Divide features into teaching data, validation data and test data in two stages
data_teaching, data_tmp, classes_teaching, classes_tmp = train_test_split(features, classes, test_size=(1/5), random_state=0)
data_test, data_validate, classes_test, classes_validate = train_test_split(data_tmp, classes_tmp, test_size=(1/2), random_state=0)

# Print teaching data, validation data and test data sample ammount
print('teaching data has {} samples'.format(classes_teaching.shape[0]))
print('validate data has {} samples'.format(classes_validate.shape[0]))
print('test data has {} samples'.format(classes_test.shape[0]))

# Perform validation of k-nearest neighbour classifier

# Loop through values from 1 to 33 with steps of 4
classification_accuracy_knn = []
k_values = range(1,34,4)
for k in k_values:
    classifier_knn = KNeighborsClassifier(n_neighbors=k).fit(data_teaching, classes_teaching)
    classification_accuracy_knn.append(accuracy_score(classes_validate, classifier_knn.predict(data_validate)))
    
# Plot a graph with different k values to see how it affects classification accuracy
plt.plot(k_values, classification_accuracy_knn)
plt.title('k-nearest neighbour classifier classification accuracy with different k values')
plt.xlabel('k value')
plt.ylabel('Classification accuracy')
plt.show()

# Finally choose best k value from graph
maximum_index_knn = np.argmax(classification_accuracy_knn)
print('Best value k value for k-nearest neighbour classifier: {}'.format(k_values[maximum_index_knn]))

# Perform validation of linear support vector machine
# Loop through exponent values of C -5 to 15
classification_accuracy_linear_svm = []
n1 = np.array(range(-5,16),dtype=float)
C1_values = 10**n1
for C1 in C1_values:
    classifier_linear_svm = LinearSVC(C=C1, random_state=0).fit(data_teaching, classes_teaching)
    classification_accuracy_linear_svm.append(accuracy_score(classes_validate, classifier_linear_svm.predict(data_validate)))
    
# Plot a graph with different exponents of C to see how it affects classification accuracy
plt.plot(n1, classification_accuracy_linear_svm)
plt.title('Linear support vector machine classification accuracy with different exponents of C')
plt.xlabel('C\'s exponent value n')
plt.ylabel('Classification accuracy')
plt.show()

# Finally choose best C value from graph
maximum_index_linear_svm = np.argmax(classification_accuracy_linear_svm)
print('Best C value for linear support vector machine: 10^({})'.format(n1[maximum_index_linear_svm]))

# Perform validation of logistic regression
# Loop through exponent values of C from -5 to 15
classification_accuracy_logistic_regression = []
n2 = np.array(range(-5,16),dtype=float)
C2_values = 10**n2
for C2 in C2_values:
    classifier_logistic_regression = LogisticRegression(C=C2, solver='sag', random_state=0).fit(data_teaching, classes_teaching)
    classification_accuracy_logistic_regression.append(accuracy_score(classes_validate, classifier_logistic_regression.predict(data_validate)))
    
# Plot a graph with different exponents of C to see how it affects classification accuracy
plt.plot(n2, classification_accuracy_logistic_regression)
plt.title('Logistic regression machine classification accuracy with different exponents of C')
plt.xlabel('C\'s exponent value n')
plt.ylabel('Classification accuracy')
plt.show()

# Finally choose best C value from graph
maximum_index_logistic_regression = np.argmax(classification_accuracy_logistic_regression)
print('Best C value for logistic regression: 10^({})'.format(n2[maximum_index_logistic_regression]))

# Teach classifier with best k, n and n2 values
#-------- Your code here --------
k = k_values[maximum_index_knn]
n1 = n1[maximum_index_linear_svm]
n2 = n2[maximum_index_logistic_regression]
#-----------------------------------

classifier_knn_validate = KNeighborsClassifier(n_neighbors=k).fit(data_teaching, classes_teaching)
classifier_linear_svm_validate = LinearSVC(C=10**(n1), random_state=0).fit(data_teaching, classes_teaching)
classifier_logistic_regression_validate = LogisticRegression(C=10**(n2), solver='sag', random_state=0).fit(data_teaching, classes_teaching)

def print_confusion_matrix(conf_matrix, title):
    """
    This function prints confusion matrix
    """
    normalized_values = []
    for row in conf_matrix:
        summ = 0
        values = []
        summ = sum(row, 0)
        for value in row:
            values.append(float(value)/float(summ))
        normalized_values.append(values)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    colors = ax.imshow(np.array(normalized_values), cmap=plt.cm.Blues, interpolation='nearest')
    width, height = conf_matrix.shape
    for i in range(width):
        for j in range(height):
            ax.annotate(str(conf_matrix[i][j]), xy=(j, i), horizontalalignment='center', verticalalignment="center")
    fig.colorbar(colors)
    classes = '0123456789'
    plt.xticks(range(width), classes[:width])
    plt.yticks(range(height), classes[:height])
    plt.title(title)
    plt.xlabel("Predicted classes")
    plt.ylabel("Correct classes")
    
# Predict classes with validated k-nearest neighbour, place predicted classes in confusion matrix and calculate classification accuracy 
classes_predicted_knn = classifier_knn_validate.predict(data_test)
print_confusion_matrix(confusion_matrix(classes_test, classes_predicted_knn), 'confusion_matrix for k-nearest neighbour classifier')
print('Classification accuracy for k-nearest neighbour: {} %'.format(round(100*accuracy_score(classes_test, classes_predicted_knn),3)))

# Predict classes with validated linear support vector machine, place predicted classes in confusion matrix and calculate classification accuracy 
classes_predicted_linear_svm = classifier_linear_svm_validate.predict(data_test)
print_confusion_matrix(confusion_matrix(classes_test, classes_predicted_linear_svm), 'confusion matrix for linear support vector machine')
print('Classification accuracy for linear support vector machine: {} %'.format(round(100*accuracy_score(classes_test, classes_predicted_linear_svm),3)))

# Predict classes with validated logistic regression, place predicted classes in confusion matrix and calculate classification accuracy 
classes_predicted_logistic_regression = classifier_logistic_regression_validate.predict(data_test)
print_confusion_matrix(confusion_matrix(classes_test, classes_predicted_logistic_regression), 'confusion matrix for logistic regression')
print('Classification accuracy for logistic regression: {} %'.format(round(100*accuracy_score(classes_test, classes_predicted_logistic_regression),3)))
plt.show()



# Predict classes with validated k-nearest neighbour, place predicted classes in confusion matrix and calculate classification accuracy 
classes_predicted_knn = classifier_knn_validate.predict(data_test)
conf_matrix_knn = confusion_matrix(classes_test, classes_predicted_knn)
print_confusion_matrix(conf_matrix_knn, 'Confusion Matrix for k-nearest neighbour classifier')
print('Classification accuracy for k-nearest neighbour: {} %'.format(round(100*accuracy_score(classes_test, classes_predicted_knn),3)))

# Get the row corresponding to the handwritten number 4 in the confusion matrix
confusion_row_4 = conf_matrix_knn[4]
# Find the index of the maximum value in the row (excluding the diagonal element)
most_confused_category_index = np.argmax(np.delete(confusion_row_4, 4))
# Print the index of the category with which the handwritten number 4 is most often confused
print('The classifier of the k-nearest neighbor most often confuses the handwritten number 4 with category:', most_confused_category_index)
