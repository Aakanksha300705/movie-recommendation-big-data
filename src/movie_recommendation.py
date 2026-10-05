from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, split, explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# --------------------------------------------------
# 1. Create Spark Session
# --------------------------------------------------

spark = SparkSession.builder \
    .appName("MovieRecommendationBigData") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# --------------------------------------------------
# 2. HDFS Paths
# --------------------------------------------------

RATINGS_PATH = "hdfs://localhost:9000/user/aakan/movielens/ratings/ratings.csv"
MOVIES_PATH = "hdfs://localhost:9000/user/aakan/movielens/movies/movies.csv"

# --------------------------------------------------
# 3. Load MovieLens Data from HDFS
# --------------------------------------------------

ratings = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(RATINGS_PATH)

movies = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(MOVIES_PATH)

print("\n========== DATASET INFORMATION ==========")
print("Total ratings:", ratings.count())
print("Total movies:", movies.count())

# --------------------------------------------------
# 4. Clean Ratings Data
# --------------------------------------------------

ratings_clean = ratings.select(
    col("userId").cast("int"),
    col("movieId").cast("int"),
    col("rating").cast("float")
).dropna()

print("\n========== CLEANED DATA ==========")
ratings_clean.printSchema()

# --------------------------------------------------
# 5. Most Rated Movies
# --------------------------------------------------

most_rated = ratings_clean.groupBy("movieId").agg(
    count("*").alias("rating_count"),
    avg("rating").alias("average_rating")
)

most_rated_movies = most_rated.join(
    movies,
    "movieId"
).orderBy(
    col("rating_count").desc()
)

print("\n========== TOP 10 MOST RATED MOVIES ==========")
most_rated_movies.select(
    "movieId",
    "title",
    "rating_count",
    "average_rating"
).show(10, truncate=False)

# --------------------------------------------------
# 6. Genre Analysis
# --------------------------------------------------

genre_ratings = movies.select(
    "movieId",
    explode(split(col("genres"), "\\|")).alias("genre")
).join(
    ratings_clean,
    "movieId"
)

genre_analysis = genre_ratings.groupBy("genre").agg(
    count("*").alias("rating_count"),
    avg("rating").alias("average_rating")
).orderBy(
    col("rating_count").desc()
)

print("\n========== GENRE ANALYSIS ==========")
genre_analysis.show(20, truncate=False)

# --------------------------------------------------
# 7. Prepare Data for ALS
# --------------------------------------------------

# Full MovieLens 25M training can exceed local Java heap.
# Therefore, a 10% sample is used for the final local ALS model.

training_small = ratings_clean.sample(
    withReplacement=False,
    fraction=0.10,
    seed=42
)

print("\nALS training records:", training_small.count())

# --------------------------------------------------
# 8. ALS Collaborative Filtering Model
# --------------------------------------------------

als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    rank=10,
    maxIter=5,
    regParam=0.1,
    coldStartStrategy="drop",
    nonnegative=True,
    seed=42
)

print("\n========== TRAINING ALS MODEL ==========")

model = als.fit(training_small)

print("ALS model training completed.")

# --------------------------------------------------
# 9. Evaluate Model
# --------------------------------------------------

test_sample = ratings_clean.sample(
    withReplacement=False,
    fraction=0.02,
    seed=123
)

predictions = model.transform(test_sample)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)

print("\n========== MODEL EVALUATION ==========")
print("RMSE:", rmse)

# --------------------------------------------------
# 10. Generate Recommendations
# --------------------------------------------------

print("\n========== RECOMMENDATIONS ==========")

user_recommendations = model.recommendForAllUsers(10)

user_31 = user_recommendations.filter(
    col("userId") == 31
)

recommendations = user_31.select(
    col("userId"),
    explode(col("recommendations")).alias("recommendation")
).select(
    col("userId"),
    col("recommendation.movieId").alias("movieId"),
    col("recommendation.rating").alias("predicted_rating")
)

recommendations_with_movies = recommendations.join(
    movies,
    "movieId"
).select(
    "userId",
    "movieId",
    "title",
    "genres",
    "predicted_rating"
).orderBy(
    col("predicted_rating").desc()
)

# Cap displayed scores at 5 for readability
recommendations_with_movies = recommendations_with_movies.withColumn(
    "predicted_rating",
    col("predicted_rating").cast("double")
)

recommendations_with_movies.show(
    10,
    truncate=False
)

# --------------------------------------------------
# 11. Example Rating Predictions
# --------------------------------------------------

print("\n========== SAMPLE PREDICTIONS ==========")

predictions.join(
    movies,
    "movieId"
).select(
    "userId",
    "movieId",
    "title",
    "rating",
    "prediction"
).show(10, truncate=False)

# --------------------------------------------------
# 12. Finish
# --------------------------------------------------

print("\n==========================================")
print("Movie Recommendation Analysis Completed")
print("==========================================")

spark.stop()
