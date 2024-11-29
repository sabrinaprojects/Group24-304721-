# Group24-304721-

## Team members
- Stefano Roscilli Zaffiri
- Elvira Francesca Tuccillo
- Sabrina Mammino

## Introduction
What does our project consist in? We have identified ourselves into memebers of the council of Marendor. Our aim is to anticipate the guild assignment for each scholar by analyzing various factors from our scholar's report: "guilds". We of course, need to be carefull cause we need to guide the kingdom's priorities.

## Methods
We have been proposed an enormous database containing "only" 31 attributes related to some features of the scholars, and 253.680 rows. Our aim is to train some models that will allow us to classify the type of guilds the scholars will be assigned to: "Apprentice", "No Guild", "MasterGuild".
The main steps we followed are:

1) LIBRARIES IMPORT
   For our project we mainly work with:
   - Panda;
   - Numpy;
   - Matplotlib.pyplot;
   - Sklearn;
   - Seaborn;

2) OSEMN (Obtain, Scrubbing, EDA, Data Modelling, Data interpretation)

   a) Obtain: we see how our databases is composed. We can already see that we have some missing values;
   
   b) Scrub/Eda (EXploratory Data Analysis):
   
   - we have plotted our data distribution based on our target attribute, and we saw that our dataset is totally **unbalanced**;

    **METTERE FOTO** ricorda di specificare il numero totale delle diverse proporzioni
   
   - We checked and plotted missing values. Here we noticed how not only our dataset was unbalanced but also full of missing values. We have an average of 25.000 missing value for each of the 31 columns.
   - Databse Size Reduction: considering how big is our dataset, in order to work with a smaller dataset that is more balanced we started by:
     
        A) Dropping all rows where the Guild Membership attribute is empty.
     
        B) Dropping all rows where (Guild Memebrship = "NO_Guild" and # empty values >= 3) and (Guild Memebrship = "Master_Guild" and # empty values >= 5) --> In this case we were more "strict" with "No_Guilds" attribute cause it is the most prominent in the database making it the main cause of our unbalancing.

     **RESULT**: we dropped a total of 119.237 rows where 113.650 belonged to "No_Guild" and 5.587 belonged to "Master Guild"

   c) DATA PREPARATION: in order to reduce the important presence of missing data that causes
   data incompleteness, algorithm inefficiency and bias we decided to use knn. However in order to run it we need to go through some Preprocessing steps:
   - ENCODING:    
   

   


































