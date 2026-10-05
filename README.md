# Large-Scale Movie Recommendation and User Preference Analysis Using Apache Spark and Hive

## Abstract

This mini project implements a large-scale movie recommendation and user preference analysis system using Apache Spark and Apache Hive. The MovieLens 25M dataset is stored and accessed through Hadoop HDFS. Apache Spark is used for large-scale data processing, genre analysis, popularity analysis, and collaborative filtering using the ALS (Alternating Least Squares) algorithm.

The system analyzes more than 25 million movie ratings and generates personalized movie recommendations based on user-item rating patterns.

## Objectives

- Process a large movie-rating dataset using Big Data technologies.
- Store and manage movie data using Hadoop HDFS.
- Use Apache Hive for structured data management.
- Perform large-scale data analysis using Apache Spark.
- Analyze movie popularity and user rating preferences.
- Apply ALS collaborative filtering for movie recommendation.
- Evaluate the recommendation model using RMSE.

## Technologies Used

- Python
- Apache Spark 4.0.1
- PySpark
- Apache Hadoop 3.4.2
- Hadoop HDFS
- Apache Hive 4.1.0
- Spark MLlib
- MovieLens 25M Dataset
- WSL2 / Ubuntu
- Matplotlib

## Dataset

The project uses the **MovieLens 25M Dataset**.

Dataset statistics:

- Movies: 62,423
- Ratings: 25,000,095
- Rating scale: 0.5 to 5.0

The dataset is not included in this repository because of its large size.

## System Architecture

```text
MovieLens 25M Dataset
          |
          v
       HDFS
          |
          +----------------+
          |                |
          v                v
       Hive             Spark
          |                |
          |       Data Cleaning & Analysis
          |                |
          |                v
          |        Genre & Popularity Analysis
          |                |
          |                v
          |          Spark MLlib ALS
          |                |
          |                v
          |       Movie Recommendations
          |                |
          v                v
       Structured       Results
        Storage       & Visualizations

Processing Workflow
Download MovieLens 25M dataset.
Store movie and rating files in Hadoop HDFS.
Create Hive tables for movies and ratings.
Load the data into PySpark from HDFS.
Clean and convert the rating columns.
Analyze movie popularity.
Analyze ratings by genre.
Train an ALS collaborative filtering model.
Generate movie recommendations.
Evaluate the model using RMSE.
Visualize the analysis results.
Big Data Analysis
Most Rated Movies

The project identifies movies with the highest number of user ratings.

Top results include:

Movie	Rating Count
Forrest Gump (1994)	81,491
Shawshank Redemption, The (1994)	81,482
Pulp Fiction (1994)	79,672
Silence of the Lambs, The (1991)	74,127
Matrix, The (1999)	72,674
Genre Analysis

The system calculates:

Total number of ratings for each genre.
Average rating for each genre.

Drama has the highest rating volume, while Film-Noir has the highest average rating among the listed genres.

Recommendation System

The recommendation system uses ALS (Alternating Least Squares) collaborative filtering from Spark MLlib.

Model parameters:

rank = 10
maxIter = 5
regParam = 0.1
coldStartStrategy = drop
nonnegative = true

The final local ALS model was trained using a 10% sample of the MovieLens ratings because training the complete dataset exceeded the available local Java heap.

Model Evaluation

The model was evaluated using Root Mean Square Error (RMSE).

RMSE = 0.9430

RMSE measures the difference between actual user ratings and ratings predicted by the recommendation model. It is not a percentage accuracy.

Example Recommendation

For a sample user, the system generates the top 10 recommended movies along with predicted ratings.

The recommendation output contains:

User ID
Movie ID
Movie title
Genres
Predicted rating
Project Structure
movie_project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── movie_recommendation.py
│
├── visualizations/
│   ├── top_10_most_rated_movies.png
│   ├── average_rating_by_genre.png
│   └── rating_volume_by_genre.png
│
└── report/
    └── Mini_Project_Report.docx
How to Run
1. Start Hadoop
start-dfs.sh
start-yarn.sh
2. Verify HDFS
hdfs dfs -ls /user/aakan/movielens
3. Start PySpark
PYSPARK_SUBMIT_ARGS="--driver-memory 4g pyspark-shell" pyspark
4. Run the Python program
spark-submit --driver-memory 4g src/movie_recommendation.py

The program loads the MovieLens data from HDFS and performs the complete analysis and recommendation workflow.

Limitations
The complete MovieLens 25M dataset requires significant memory for ALS training.
The final local ALS model therefore uses a 10% sample.
Hive is mainly used for structured data management, while Spark performs the main large-scale analytics and machine learning.
The dataset itself is not included in this GitHub repository.
Future Scope
Train the recommendation model on a distributed Spark cluster.
Improve recommendation quality through hyperparameter tuning.
Add a web-based recommendation interface.
Include content-based filtering using movie genres.
Combine collaborative filtering and content-based filtering into a hybrid recommendation system.
Add real-time recommendation capabilities.
Conclusion

The project demonstrates how Big Data technologies can be used to analyze large-scale movie-rating data and build a recommendation system. Hadoop HDFS provides distributed storage, Hive provides structured data management, and Apache Spark performs large-scale processing and machine learning. The ALS collaborative filtering model successfully generates personalized movie recommendations, while the additional analyses provide insights into movie popularity and user preferences.

Author

Aakanksha Sherkar

Department of Computer Engineering
SIES Graduate School of Technology
Navi Mumbai
