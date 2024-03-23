from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures



def glaucoma_classification(embeds, labs, train_ratio=0.7):
    
    train_x, test_x, train_y, test_y = train_test_split(embeds, labs, test_size=(1-train_ratio), random_state=42)

    classifier = LinearSVC()
    hisroty = classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)
    
    Acc = accuracy_score(test_y, y_pred)
    
    fpr, tpr, thresholds = roc_curve(test_y, y_pred)
    AUC = auc(fpr, tpr)
    
    return Acc, AUC


def meanVisualField_prediction(embeds, mds, train_ratio=0.7):
    
    train_x, test_x, train_y, test_y = train_test_split(embeds, mds, test_size=(1-train_ratio), random_state=42)

    regressor = LinearRegression()
    
    hisroty = regressor.fit(train_x, train_y)
    y_pred = regressor.predict(test_x)

    MAE = mean_absolute_error(test_y, y_pred)
    R2 = r2_score(test_y, y_pred)
    
    return MAE, R2