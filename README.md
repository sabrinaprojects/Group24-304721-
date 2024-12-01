# Group24

## Team Members
- Stefano Roscilli Zaffiri (304721)  
- Elvira Francesca Tuccillo (305151)  
- Sabrina Mammino (307371)
  
## Introduction
What does our project consist of? We have identified ourselves as members of the council of Marendor. Our aim is to anticipate the guild assignment for each scholar by analyzing various factors from the scholars' reports: "guilds." We, of course, need to be careful because we must guide the kingdom's priorities.

## Methods
We were given an enormous database containing 31 attributes related to some features of the scholars, with a total of 253,680 rows. Our goal is to train models to classify the type of guilds the scholars will be assigned to: "Apprentice_Guild", "No Guild," or "Master Guild."  
The main steps we followed are:

### 1) Libraries Import
For our project, we mainly worked with the following libraries:  
- Pandas  
- Numpy  
- Matplotlib.pyplot  
- Sklearn  
- Seaborn  

### 2) OSEMN (Obtain, Scrubbing, EDA, Data Modeling, Data Interpretation)

#### A) Obtain  
We started by reading our CSV file and examining the composition of our database. By looking at some samples, we observed that our dataset contained a mix of numerical and categorical columns. We also noted that we will have to deal with missing values.

#### B) Scrubbing/Exploratory Data Analysis (EDA)  
1. *Distribution of Variables:*  
   - We plotted histograms of the data to analyze the distribution of all variables, emphasizing our target attribute (Guild_Membership). From this, we saw that our dataset is highly *unbalanced*.  

   ![Distribution of Classes!](images/Distribution.png 'Distribution ')

2. *Missing Values:*  
   - We checked and plotted the missing values. We found that our dataset not only had a severe class imbalance but also contained a significant number of missing values. On average, there were approximately 25,000 missing values for each of the 31 columns.

3. *Database Size Reduction:*  
Considering the size of our dataset and the need to remove the imbalance, we followed these steps to resize the data:

1. **Dropped all rows where the target Guild_Membership attribute was empty.**
2. *Dropped rows based on missing values thresholds:*
   - For Guild_Membership = "Master_Guild", rows with 5 or more missing values were removed.
   - For Guild_Membership = "No_Guild", rows with 3 or more missing values were removed.
   - This stricter threshold for No_Guild was necessary because it was the most prominent class, contributing significantly to the imbalance.

#### Result:
We dropped a total of *119,237 rows*, including:
- *113,650 rows* from the No_Guild class
- *5,587 rows* from the Master_Guild class

Before settling on these thresholds, we experimented with different combinations. Some of the combinations we tested are:
- No_Guild = 3 and Master_Guild = 4
- No_Guild = 4 and Master_Guild = 5
- No_Guild = 4 and Master_Guild = 6

After several trials, we found that each combination either was too strict with one class and too loose with the other. Ultimately, the combination of **3 for No_Guild** and **5 for Master_Guild** provided the best balance, so we decided to use it.


#### C) Data Preparation  
To address the significant presence of missing data (which could cause algorithm inefficiency and bias), we used the KNN imputer. However, we first needed to preprocess the data through these steps:  

1. *Encoding:*  
   - For the target column (Guild_Membership), we mapped the classes as follows:  
     - No_Guild = 0  
     - Master_Guild = 1  
     - Apprentice_Guild = 2  
   - For other categorical columns (binary values: 'Absent' or 'Present'), we used the Label Encoder:  
     - Absent = 0  
     - Present = 1  
   - Since Label Encoders do not handle missing values (NaN), we temporarily replaced missing values with a placeholder, applied the encoding, and then substituted the placeholder back with NaN.

2. *Outlier Handling:*  
   - We identified potential outliers using boxplots for numerical columns (excluding binary categorical columns because it would be useless to boxplot them since they do not present outliers). Many columns contained a significant number of outliers.  
   - To handle this, we analyzed the value distributions using histograms and removed outliers based on each column's specific characteristics. This approach ensured minimal loss of valuable information.  

3. Correlation Analysis  

- We computed and visualized the correlation matrix for our dataset to assess relationships between variables and the target variable (`Guild_Membership`). 
- To determine which features to keep, we tested different correlation thresholds, including 0.1, 0.08, 0.07, and 0.06.  
- Based on these experiments, we decided to drop columns with an absolute correlation value < 0.07 with the target variable. This threshold provided the optimal balance between maintaining useful features and dropping useless ones.  
- By removing low-correlation variables that did not contribute meaningful information, we reduced the dataset's size and improved model performance.

-Correlation matrix after the drop:

  ![Correlation Matrix!](images/Correlation_Matrix.png 'Correlation Matrix ')

4. *Feature Scaling:*  
   - We firstly tried scaling with the *Standard Scaler* but it worked poorly since we had some outliers left from the cleaning process. So we scaled the features using the *Robust Scaler* because it handles outliers effectively. This ensured that any leftover outliers was taken care of. 

5. *Dataset Balancing:*  
   - To create a smaller, more balanced dataset, we included:  
     - All rows with Guild_Membership = "Apprentice_Guild" (4.102 rows)  
     - 6.000 rows with Guild_Membership = "Master_Guild"  
     - 10.000 rows with Guild_Membership = "No_Guild"  
   - This rebalanced the dataset while maintaining a similar class distribution to the original data, reducing bias.
   - Before settling on the final sampling (4,102, 6,000, and 10,000), we tested other distributions, such as 3,000, 5,000, and 10,000, or 4,102, 10,000, and 20,000. We did this to explore different splits while maintaining, as much as possible, the original distribution. However, the other samples resulted in poor performance, particularly for the 'Apprentice_Guild' class. As a result, we decided to stick with the sampling that gave the best detection for that class.

6. *KNN Imputation:*  
   - Finally, we applied the KNN imputer, filling the missing values by considering the 5 nearest neighbors.
   - Before deciding to use the KNN imputation we also tried other methods like the mean and median for numerical columns and the mode for categorical ones. However these methods were not good because when we tried them they would fill too many values with the same input leading to a huge difference in the distribution of all the variables 

#### D) Data Modeling  
1. *Model Selection:*  
   We selected models suitable for classification problems:  
   - Logistic Regression  
   - Kernel Support Vector Machines (KSVM)  
   - CART Trees  
   - Random Forest  

2. *Stratified Splitting:*

   - We tried splitting in two different ways: first we try 80% train set and 20% test set and then we try 70% train set and 30% test set andd we observe that our models work better with the 70-30 split so we decide to:
     
       - split the data into training (70%) and test (30%) sets, maintaining the same class distribution thanks to stratification.
       
       - The training set was further divided into training (70%) and validation (30%) subsets for model evaluation.

3. *Evaluation Metrics:*  
   - We used the following metrics to evaluate model performance:  
     - *Accuracy:* Proportion of correctly predicted instances for every class (correct prediction of 0, 1 and 2).  
     - *Precision:* Proportion of true positives among predicted positives for each class.  
     - *Recall:* Proportion of true positives among actual instances of each class.  
     - *F1-Score:* Harmonic mean of precision and recall (detecting their trade-off) for all specific classes.  
     - *Macro-Average and Weighted-Average* (for precision, recall, and F1-score).  
   - For Logistic Regression and KSVM, we also analyzed the:  
     - *ROC Curve* plots the trade-off of Recall (Y-axis) vs. False Positive Rate (X-axis) (for ecample if we look at FPR for class 0 it will return how many instances of class 1 and class 2 are detected as 0)
     - *ROC AUC (Area Under the ROC Curve):* Higher values indicate better model performance.

4. *Model Performance:*  
   - *CART Trees:* The first model we tried was Cart Trees and it had really bad metrics overall.
     
     ![Cart Trees Performance!](images/Cart_Trees_Perf.png 'Carte Trees Performance ')
   
   - *Random Forest:* It had much better metrics then Cart Trees but struggled with class 2 detection.
     
     ![Random Forest Performance!](images/Random_Forest_Perf.png 'Random Forest Performance ')
     
   - *Logistic Regression:* Was very similar to Random Forest but a little worse for class 2.
     
     ![Logistic Regression Performance!](images/Logistic_Regression_Perf.png 'Logistic Regression Performance ')
     
   - *KSVM:* Performed worse than Logistic Regression and Random Forest but better than CART Trees overall.
     
     ![KSVM Performance!](images/KSVM_Perf.png 'KSVM Performance ')

#### E) Results  
1. *Hyperparameter Tuning:*  
   - We used Grid Search to find the optimal hyperparameters for our best models: Random Forest and Logistic Regression, testing various combinations to maximize performance.  

2. *Test Set Results:*  
   - We then tried Random Forest and Logistic Regression on the test set created before and they showed similar overall performance.  
     - Logistic Regression excelled in *Recall* for classes 0 and 2.  
     - Random Forest achieved higher *Precision.*  
     - Both models performed similarly for class 1.
    
       
     -Random Forest
       
       ![Best RF on Test set!](images/Best_RF_Test.png 'Best RF on test set ')
    
     -Logistic Regression
     
       ![Best LR on test set!](images/Best_LR_Test.png 'Best LR on test set ')
    
### F) Full Dataset  
After training and testing our models on the smaller sample of the dataset, we extended the process to the full dataset.

1. **Filling Missing Values:**  
   - Initially, we attempted to fill missing values using the **KNN imputer** with 5 nearest neighbors. However, this approach proved computationally expensive and time-consuming.  
   - To address this, we reduced the number of neighbors to 3, significantly improving processing time while still effectively handling the missing values.

2. **Model Testing:**  
   - We then applied our trained and tuned models to the full dataset.  
   - The performance of our models improved drastically compared to the sample dataset.  
   - Despite this improvement, challenges remained with class 2 (`Apprentice_Guild`) due to the severe imbalance in the original dataset.

3. **Performance Comparison:**  
   - Analyzing the performance reports of **Random Forest** and **Logistic Regression**, we observed that **Random Forest** performed better overall on the full dataset.
   - Random Forest:
     
     ![Best RF on total df!](images/Best_RF_total.png 'Best RF on total df ')
  
   - Logistic Regression:
     
     ![Best LR on total df!](images/Best_LR_total.png 'Best LR on test set ')

**So the winner is...** 🎉 **Random Forest!**

---

### Conclusion  
On the sampled dataset, **Random Forest** and **Logistic Regression** were equally strong candidates, each excelling in different areas:  
- For **higher recall** on classes 0 (`No_Guild`) and 2 (`Apprentice_Guild`), we recommend **Logistic Regression**.  
- For **higher precision**, **Random Forest** is the better choice.  
- Both models performed comparably on class 1 (`Master_Guild`).  

However, when tested on the entire dataset, **Random Forest** consistently outperformed Logistic Regression. As a result, **Random Forest** emerges as the optimal choice for this task.
