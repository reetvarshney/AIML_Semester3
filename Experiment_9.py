# Importing necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek
from collections import Counter

# Generate synthetic imbalanced data
X, y = make_classification(n_samples=5000, n_features=10, n_classes=2, 
                           weights=[0.9, 0.1], random_state=42)

# Check class distribution before handling imbalance
print("Class distribution before handling imbalance: ", Counter(y))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Random Over-Sampling
def random_over_sampling():
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    print("Class distribution after Random Over-Sampling: ", Counter(y_res))
    
    # Training the classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    print("\nRandom Over-Sampling Classification Report:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# 2. Random Under-Sampling
def random_under_sampling():
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print("Class distribution after Random Under-Sampling: ", Counter(y_res))
    
    # Training the classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    print("\nRandom Under-Sampling Classification Report:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# 3. SMOTE (Synthetic Minority Over-sampling Technique)
def smote_sampling():
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE: ", Counter(y_res))
    
    # Training the classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    print("\nSMOTE Sampling Classification Report:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# 4. NearMiss (Under-sampling method)
def nearmiss_sampling():
    nearmiss = NearMiss()
    X_res, y_res = nearmiss.fit_resample(X_train, y_train)
    print("Class distribution after NearMiss: ", Counter(y_res))
    
    # Training the classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    print("\nNearMiss Sampling Classification Report:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# 5. SMOTETomek (Combining SMOTE and Tomek Links)
def smote_tomek_sampling():
    smote_tomek = SMOTETomek(random_state=42)
    X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
    print("Class distribution after SMOTETomek: ", Counter(y_res))
    
    # Training the classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    print("\nSMOTETomek Sampling Classification Report:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# 6. Using Class Weights in Random Forest Classifier
def class_weight_model():
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    print("\nClass Weight Model Classification Report:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# Running different techniques to handle imbalanced data
if _name_ == "_main_":
    print("Before Handling Imbalance:")
    print(Counter(y_train))
    
    print("\n--- Random Over-Sampling ---")
    random_over_sampling()
    
    print("\n--- Random Under-Sampling ---")
    random_under_sampling()
    
    print("\n--- SMOTE ---")
    smote_sampling()
    
    print("\n--- NearMiss ---")
    nearmiss_sampling()
    
    print("\n--- SMOTETomek ---")
    smote_tomek_sampling()
    
    print("\n--- Class Weight in Random Forest ---")
    class_weight_model()