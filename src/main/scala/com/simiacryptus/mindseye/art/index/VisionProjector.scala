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

import java.awt.Desktop
import java.net.{InetAddress, URI}
import java.util.UUID

import com.fasterxml.jackson.annotation.JsonIgnore
import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.index.VisionProjector._
import com.simiacryptus.mindseye.art.models.VGG19
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util.{ArtSetup, BasicOptimizer, VisionPipelineUtil}
import com.simiacryptus.mindseye.art.{VisionPipeline, VisionPipelineLayer}
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer
import com.simiacryptus.mindseye.layers.java.ImgTileAssemblyLayer
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.{AWSNotebookRunner, EC2Runner, NotebookRunner}
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

import scala.collection.JavaConversions._
import scala.util.{Random, Try}


object VisionProjector_EC2 extends VisionProjector with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def urlBase: String = String.format("http://%s:1080/etc/", InetAddress.getLocalHost.getHostAddress)

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def spark_master = "local[1]"

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> spark_master
  )

}
object VisionProjector_Local extends VisionProjector with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5

  override def spark_master = "local[1]"

  override val urlBase: String = "http://localhost:1080/etc/"

}

abstract class VisionProjector extends ArtSetup[Object] with BasicOptimizer {

  @JsonIgnore def spark_master: String

  @JsonIgnore def sparkFactory: SparkSession = {
    val builder = SparkSession.builder()
    import scala.collection.JavaConverters._
    VisionPipelineUtil.getHadoopConfig().asScala.foreach(t => builder.config(t.getKey, t.getValue))
    builder.master("local[1]").getOrCreate()
  }

  val archiveUrl = "file:///C:/Users/andre/data/images/"
  val inputUrl = "s3a://data-cb03c/crawl/wikiart/"
  val toIndex = 100
  val toDisplay = 1000
  val indexResolution = 512
  val thumbnailResolution = 128
  val queries = Map(
    "self-portrait-with-bandaged-ear-1889.jpg!Large.jpg" -> "VGG19_1d4",
    "the-beloved-toy-2006.jpg!Large.jpg" -> "VGG19_1d4",
    "prince-muhammad-beik-of-georgia-1620.jpg!Large.jpg" -> "VGG19_1d4"
  )

  def urlBase: String

  override def cudaLog = false

  override def postConfigure(log: NotebookOutput) = {
    implicit val sparkSession = sparkFactory
    val files: Array[String] = findFiles(Set(
      "Large"
    ), base = inputUrl).filter(!_.contains("Small"))
    val index = sparkSession.read.parquet(archiveUrl).cache()
    for((example, matcher) <- queries) {
      log.subreport(example.split('.').head, (sub:NotebookOutput)=>{
        val keyRow = index.where(index("file").contains(lit(example)).and(
          index("layer").eqNullSafe(lit(matcher))
        )).head()
        val dataframe = select(index, keyRow, toDisplay)
        val fileKeys = dataframe.select("file").rdd.map(_.getString(0)).distinct().collect()
        val pngFile: String = getThumbnailImage(fileKeys, thumbnailResolution)(sub)
        val embeddingConfigs = for (Row(pipeline: String, resolution: Int, layer: String) <- dataframe.select("pipeline", "resolution", "layer").distinct().collect()) yield {
          val label = s"${pipeline}_${layer}_$resolution"
          sub.h1(label)
          val embeddings = dataframe.where((dataframe("pipeline") eqNullSafe pipeline) and (dataframe("resolution") eqNullSafe resolution) and (dataframe("layer") eqNullSafe layer))
            .select("file", "data").limit(toDisplay).collect()
            .map({ case Row(file: String, data: Seq[Double]) => file -> data.toArray }).toMap
            .mapValues(data => new Tensor(data, data.length))

          val config = getProjectorConfig(label, embeddings, urlBase, thumbnailResolution, fileKeys, pngFile)(sub)
          logEmbedding(urlBase, config)(sub)
          config
        }
        displayEmbedding(urlBase, embeddingConfigs: _*)(sub)
        null
      })
    }
    null
  }
  def select(index: DataFrame, exampleRow: Row, window:Int)(implicit sparkSession: SparkSession):DataFrame = {
    val files = index.where(index("resolution").eqNullSafe(exampleRow.getAs[Int]("resolution")).and(
      index("pipeline").eqNullSafe(exampleRow.getAs[String]("pipeline"))
    ).and(
      index("layer").eqNullSafe(exampleRow.getAs[String]("layer"))
    )).rdd.sortBy(r => r.getAs[Seq[Double]]("data").zip(exampleRow.getAs[Seq[Double]]("data"))
      .map(t => t._1 - t._2).map(x => x * x).sum).map(_.getAs[String]("file")).distinct.take(window).toSet
    index.filter(r=>files.contains(r.getAs[String]("file")))
  }

}

object VisionProjector {

  def indexImages(visionPipeline: => VisionPipeline[VisionPipelineLayer], toIndex: Int, indexResolution: Int, archiveUrl:String)
                 (files: String*)
                 (implicit sparkSession: SparkSession): DataFrame = {
    val indexed = Try {
      val previousIndex = sparkSession.read.parquet(archiveUrl)
        .where((col("pipeline") eqNullSafe lit(visionPipeline.name)) and (col("resolution") eqNullSafe indexResolution))
      previousIndex.select("file").rdd.map(_.getString(0)).distinct().collect().toSet
    }.getOrElse(Set.empty)
    println("Current files in index: " + indexed.size)
    val allFiles = Random.shuffle(files.filter(!indexed.contains(_)).toList).distinct.take(toIndex)
    if(allFiles.isEmpty) sparkSession.emptyDataFrame
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

  def getProjectorConfig(name: String, embeddings: Map[String, Tensor], urlBase: String, spriteSize: Int, images: Array[String], pngFile: String)
                                (implicit log: NotebookOutput)= {
    val id = UUID.randomUUID().toString
    log.p(log.file(images.map(embeddings).map(_.getData.map("%.4f".format(_)).mkString("\t")).mkString("\n"), s"$id.tensors.tsv", "Data"))
    log.p(log.file((
      List(List("Label", "URL").mkString("\t")) ++
        images.map(img => List(
          img.split('/').last,
          img.replaceAll("s3a://simiacryptus/", "https://simiacryptus.s3.amazonaws.com/")
            .replaceAll("s3a://", "https://s3.amazonaws.com/")
        ).mkString("\t"))
      ).mkString("\n"), s"$id.metadata.tsv", "Metadata"))
    Map(
      "tensorName" -> name,
      "tensorShape" -> Array(images.size, embeddings.head._2.length()),
      "tensorPath" -> s"$urlBase$id.tensors.tsv",
      "metadataPath" -> s"$urlBase$id.metadata.tsv",
      "sprite" -> Map(
        "imagePath" -> s"$urlBase$pngFile",
        "singleImageDim" -> Array(spriteSize, spriteSize)
      )
    )
  }

  def getThumbnailImage(images: Array[String], spriteSize: Int)
                       (implicit log: NotebookOutput) = {
    val rowCols = Math.sqrt(images.length).ceil.toInt
    val sprites = new ImgTileAssemblyLayer(rowCols, rowCols).eval(
      (images ++ images.take((rowCols * rowCols) - images.length)).par.map(VisionPipelineUtil.load(_, spriteSize, spriteSize)).map(Tensor.fromRGB(_)).toArray: _*
    ).getDataAndFree.getAndFree(0)
    val pngTxt = log.png(sprites.toRgbImage, "sprites")
    log.p(pngTxt)
    val pngFile = pngTxt.split('(').last.stripSuffix(")").stripPrefix("etc/")
    pngFile
  }


  def displayEmbedding(urlBase: String, embedding: Map[String, Any]*)
                      (implicit log: NotebookOutput) = {
    val projectorUrl: String = logEmbedding(urlBase, embedding: _*)
    Try {
      Desktop.getDesktop.browse(new URI(projectorUrl))
    }
    embedding
  }

  def logEmbedding(urlBase: String, embedding: Map[String, Any]*)
                  (implicit log: NotebookOutput) = {
    val id = UUID.randomUUID().toString
    log.p(log.file(ScalaJson.toJson(Map("embeddings" -> embedding.map(identity).toArray)), s"$id.json", "Projector Config"))
    val projectorUrl = s"""https://projector.tensorflow.org/?config=$urlBase$id.json"""
    log.p(s"""<a href="$projectorUrl">Projector</a>""")
    projectorUrl
  }
}