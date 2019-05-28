/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art.index

import java.net.InetAddress

import com.fasterxml.jackson.annotation.JsonIgnore
import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.index.VisionIndexer._
import com.simiacryptus.mindseye.art.models.VGG19
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util.{ArtSetup, BasicOptimizer, VisionPipelineUtil}
import com.simiacryptus.mindseye.art.{VisionPipeline, VisionPipelineLayer}
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner
import com.simiacryptus.sparkbook.{AWSNotebookRunner, EC2Runner, NotebookRunner}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

import scala.collection.JavaConversions._
import scala.util.{Random, Try}


object VisionIndexer_EC2 extends VisionIndexer with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def urlBase: String = String.format("http://%s:1080/etc/", InetAddress.getLocalHost.getHostAddress)

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> spark_master
  )

  override def spark_master = "local[1]"

}

object VisionIndexer_Local extends VisionIndexer with LocalRunner[Object] with NotebookRunner[Object] {
  override val urlBase: String = "http://localhost:1080/etc/"

  override def inputTimeoutSeconds = 5

  override def spark_master = "local[1]"

}


abstract class VisionIndexer extends ArtSetup[Object] with BasicOptimizer {

  val archiveUrl = "file:///C:/Users/andre/data/images/"
  val inputUrl = "s3a://data-cb03c/crawl/wikiart/"
  val toIndex = 100
  val toDisplay = 1000
  val indexResolution = 512
  val thumbnailResolution = 128

  @JsonIgnore def spark_master: String

  def urlBase: String

  override def cudaLog = false

  override def postConfigure(log: NotebookOutput) = {
    implicit val sparkSession = sparkFactory
    val files: Array[String] = findFiles(Set(
      "Large"
    ), base = inputUrl).filter(!_.contains("Small"))
    val index = indexAll(files, indexImages(VGG19.getVisionPipeline, toIndex, indexResolution, archiveUrl))
    null
  }

  @JsonIgnore def sparkFactory: SparkSession = {
    val builder = SparkSession.builder()
    import scala.collection.JavaConverters._
    VisionPipelineUtil.getHadoopConfig().asScala.foreach(t => builder.config(t.getKey, t.getValue))
    builder.master("local[1]").getOrCreate()
  }

  def indexAll(files: Array[String], indexer: Seq[String] => DataFrame)(implicit sparkSession: SparkSession): DataFrame = {
    var dataframe: DataFrame = null
    while (dataframe == null || !dataframe.isEmpty) {
      dataframe = indexer(files)
      if (!dataframe.isEmpty) dataframe.coalesce(files.size / 100).write.mode(SaveMode.Append).parquet(archiveUrl)
    }
    val frame = sparkSession.read.parquet(archiveUrl).cache()
    println("Index contains " + frame.select("file").distinct().count() + " files")
    frame
  }

}

object VisionIndexer {

  def indexImages(visionPipeline: => VisionPipeline[VisionPipelineLayer], toIndex: Int, indexResolution: Int, archiveUrl: String)
                 (files: String*)
                 (implicit sparkSession: SparkSession): DataFrame = {
    val indexed = Try {
      val previousIndex = sparkSession.read.parquet(archiveUrl)
        .where((col("pipeline") eqNullSafe lit(visionPipeline.name)) and (col("resolution") eqNullSafe indexResolution))
      previousIndex.select("file").rdd.map(_.getString(0)).distinct().collect().toSet
    }.getOrElse(Set.empty)
    println("Current files in index: " + indexed.size)
    val allFiles = Random.shuffle(files.filter(!indexed.contains(_)).toList).distinct.take(toIndex)
    if (allFiles.isEmpty) sparkSession.emptyDataFrame
    else index(visionPipeline, indexResolution, allFiles: _*)
  }

  def index(pipeline: => VisionPipeline[VisionPipelineLayer], imageSize: Int, images: String*)
           (implicit sparkSession: SparkSession) = {
    val rows = sparkSession.sparkContext.parallelize(images, images.length).flatMap(file => {
      val layers = pipeline.getLayers.toArray
      val canvas = Tensor.fromRGB(VisionPipelineUtil.load(file, imageSize))
      val tuples = layers.foldLeft(List(canvas))((input, layer) => {
        val l = layer._1.getLayer
        val tensors = input ++ List(l.eval(input.last).getDataAndFree.getAndFree(0))
        l.freeRef()
        tensors
      })
      tuples.head.freeRef()
      val reducerLayer = new BandAvgReducerLayer()
      val rows = (layers.map(_._1.name()) zip tuples.tail).toMap
        .mapValues(data => {
          val tensor = reducerLayer.eval(data).getDataAndFree.getAndFree(0)
          data.freeRef()
          val doubles = tensor.getData.clone()
          tensor.freeRef()
          doubles
        }).map(t => {
        Array(file, imageSize, pipeline.name, t._1, t._2)
      })
      reducerLayer.freeRef()
      println(s"Indexed $file")
      rows
    }).cache()
    sparkSession.createDataFrame(rows.map(Row(_: _*)), StructType(Array(
      StructField("file", StringType),
      StructField("resolution", IntegerType),
      StructField("pipeline", StringType),
      StructField("layer", StringType),
      StructField("data", ArrayType(DoubleType))
    )))
  }


}