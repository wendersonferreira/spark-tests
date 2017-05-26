import scala.collection.JavaConversions._
import scala.io.Source
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by Wenderson Ferreira de Souza on 26/05/17.
  */
object IrisKMeans {
  def main(args: Array[String]) {

    val appName = "IrisKMeans"
    val master = "local"
    val conf = new SparkConf().setAppName(appName).setMaster(master)
    val sc = new SparkContext(conf)

    println("Loading iris data from URL...")
    val url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    val src = Source.fromURL(url).getLines.filter(_.nonEmpty).toList
    val textData = sc.parallelize(src)
    val parsedData = textData
      .map(_.split(",").dropRight(1).map(_.toDouble))
      .map(Vectors.dense).cache()

    val numClusters = 3
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)
  }
}