import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#import training data
train_data = pd.read_csv('/Users/georgizahariev/Desktop/GMU/5 - Fall 2024/CS484/HW2/1726777589_1235836_cleveland-train.csv')

#printing head, just to see that I am reading the file properly
print(train_data.head())

#using pandas iloc I slice the traing data - into X train data and Y train labels
#this makes X_train all the colomns without the last one
X_train = train_data.iloc[:, :-1].values
#makes Y_train - just the last colomn
Y_train = train_data.iloc[:, -1].values

# next step is data preprocessing 

scaler = StandardScaler()

#scaling - normalizing the data 
#transforming the data as well
X = scaler.fit_transform(X_train)

def sigmoid(z):
    #using formula
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y, y_pred):
    #using formula
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def classification_error(y, y_pred):
    # Convert probabilities to 1 or 0
    y_pred_labels = np.where(y_pred >= 0.5, 1, 0)
    
    # Classification error is 1 - accuracy
    return 1 - accuracy_score(y, y_pred_labels)

# gradient descent function, it needs 2 required parameters and 3 optional
#I have put the values of the optional as in the HW description
#max_iter is being set to different values when I am testing the program, but I decided toput default value of 10 thousands
def logistic_regression_gradient_descent(X, y, learning_rate=0.00001, max_iters=10000, tolerance=0.001):
    # Add a column of ones
    X = np.c_[np.ones(X.shape[0]), X]
    # Initialize weights
    w = np.zeros(X.shape[1])
    
    start_time = time.time()  #start tracking the time for the number of operations
    
    for i in range(max_iters):
        z = np.dot(X, w)  # dot product - linear combination between X's values and weights's values
        y_pred = sigmoid(z)  # Sigmoid output for our logistic regression
        loss = cross_entropy_loss(y, y_pred)  # cross-entropy loss
        gradient = np.dot(X.T, (y_pred - y)) / y.size  # Computing gradient

        w = w - (learning_rate * gradient)  # Update weights
        
        # Check if converge, | gradient | < 0.001, => stop the cycle
        if np.all(np.abs(gradient) < tolerance):
            print(f"Converged after {i} iterations.")
            break
    
    end_time = time.time()
    #stop tracking the time
    training_time = end_time - start_time  # total time
    error = classification_error(y, y_pred)  # training error / classification error
    
    #returning weight vector, cross entropy loss and classification error as explained in the description
    #moreover, returning the training timeas needed later in the HW implementation
    return w, loss, error, training_time

# the needed iterations from Project description
max_iters_list = [1000000, 100000, 10000]

# adapt Y-train => making him 0 and 1
Y_train = np.where(Y_train == -1, 0, Y_train)

# run the algorithm with the max iteration values
# returning the value with self explanatory text -so whoever runs will understand what is the meaning of te output
for max_iters in max_iters_list:
    print(f"\nRunning model with max_iters = {max_iters}")
    weights, loss, error, train_time = logistic_regression_gradient_descent(X, Y_train, max_iters=max_iters)
    
    # Report the results for the training set
    print(f"Training Cross-Entropy Loss: {loss}")
    print(f"Training Classification Error: {error}")
    print(f"Training Time: {train_time:.2f} seconds")


#reading the test data
X_test = pd.read_csv('/Users/georgizahariev/Desktop/GMU/5 - Fall 2024/CS484/HW2/1726777589_1241634_cleveland-test.csv').values

# preprocessing and trasforming the testing data
X_testing = scaler.transform(X_test)
X_test_scaled_intercept = np.c_[np.ones(X_testing.shape[0]), X_testing]

# Make predictions using the learned weights
y_test_pred = sigmoid(np.dot(X_test_scaled_intercept, weights))

#adjust the result fto -1/+1 , as that's the format we need
y_test_pred_labels = np.where(y_test_pred >= 0.5, 1, -1)

#During testing I had floating points in mine -1/1 
# and I c to integers to avoid float representations
y_test_pred_labels = y_test_pred_labels.astype(int)

# Save the predictions to a CSV file with integer formatting => -1/1
np.savetxt('test_predictions_HW2.csv', y_test_pred_labels, fmt='%d', delimiter=',')
print("Test predictions saved to 'test_predictions_HW2.csv'\n")

#.......................................................................
#Now I will run the logistic regression using sklearn library
# the output files will be compared on Miner to see which one is more accurate
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# earlier I changed Y_train to be 0 and 1, now I am changing it back to its initial state -1 and 1
Y_train = np.where(Y_train == 0, -1, Y_train)

#recording the time, needed for training
start_time = time.time()

model.fit(X, Y_train)
end_time = time.time()

train_time = end_time - start_time
print(f"Training Time: {train_time:.2f} seconds\n")

X_test2 = scaler.transform(X_test)

test_predictions = model.predict(X_test2)

# Saving the results of my prediction to a .dat file
with open('result_HW2.csv', 'w') as f:
    for prediction in test_predictions:
        f.write(f'{prediction}\n')

print("Test predictions saved to 'result_HW2.csv'\n")
    