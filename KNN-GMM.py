# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql import SQLContext

spark = SparkSession.builder.appName("CommonFriends").getOrCreate()
iris_csv=spark.read.option("header",True).csv("/FileStore/tables/Iris.csv")
display(iris_csv)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
iris_data=iris_csv.drop("Id")
iris_data=iris_data.drop("Species")
iris_data=iris_data.withColumn("SepalLengthCm",col("SepalLengthCm").cast(DoubleType()))\
    .withColumn("SepalWidthCm",col("SepalWidthCm").cast(DoubleType()))\
    .withColumn("PetalLengthCm",col("PetalLengthCm").cast(DoubleType()))\
    .withColumn("PetalWidthCm",col("PetalWidthCm").cast(DoubleType()))

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
assemble=VectorAssembler(inputCols=iris_data.columns,outputCol = 'Features')
assembled_data=assemble.transform(iris_data)

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
silhouette_scores=[]
evaluator = ClusteringEvaluator(featuresCol='Features', metricName='silhouette', distanceMeasure='squaredEuclidean')
for K in range(2,11):
    k_means= KMeans(featuresCol='Features',k=K)
    k_means_fit=k_means.fit(assembled_data)
    k_means_transform=k_means_fit.transform(assembled_data) 
    evaluation_score=evaluator.evaluate(k_means_transform)
    silhouette_scores.append(evaluation_score)

# COMMAND ----------

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize =(10,8))
ax.plot(range(2,11),silhouette_scores,linestyle='--', marker='o',color="red")
ax.set_xlabel('K')
ax.set_ylabel('Silhouette_Score')
plt.show()

# COMMAND ----------

from pyspark.ml.clustering import GaussianMixture
K=2
gm= GaussianMixture(k=K,featuresCol="Features")
gm_fit=gm.fit(assembled_data)
gm_transform=gm_fit.transform(assembled_data) 
evaluation_score=evaluator.evaluate(gm_transform)
print(f"Silhouette Score using Guassian Mixture Model for K={K}: {evaluation_score}")

# COMMAND ----------

K=2
kmeans= KMeans(k=K,featuresCol="Features")
kmeans_fit=kmeans.fit(assembled_data)
kmeans_transform=kmeans_fit.transform(assembled_data) 
evaluation_score=evaluator.evaluate(kmeans_transform)
print(f"Silhouette Score using K-Means for K={K}: {evaluation_score}")
