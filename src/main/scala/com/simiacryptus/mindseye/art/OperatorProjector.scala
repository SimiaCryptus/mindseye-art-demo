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

package com.simiacryptus.mindseye.art

import java.awt.Desktop
import java.net.URI
import java.util.UUID

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.models.VGG19
import com.simiacryptus.mindseye.art.ops._
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util._
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.layers.java.ImgTileAssemblyLayer
import com.simiacryptus.mindseye.opt.TrainingMonitor
import com.simiacryptus.notebook.{NotebookOutput, TableOutput}
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.util.{Random, Try}

object OperatorProjector_EC2 extends OperatorProjector with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object OperatorProjector_Local extends OperatorProjector with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5

  override def s3bucket: String = ""
}

abstract class OperatorProjector extends ArtSetup[Object] with BasicOptimizer {

  val styleMagnification = 1.0
  val styleMin = 64
  val styleMax = 1280
  val stylePixelMax = 1e7
  val spriteSize = 128
  val images = Random.shuffle(List(
    "vincent-van-gogh",
    "m-c-escher",
    "salvador-dali",
    "leonardo-da-vinci",
    "norman-rockwell",
    "pablo-picasso",
    "/michelangelo/"
  ).map(str => str -> findFiles(str, base = "s3a://data-cb03c/crawl/wikiart/images/").filter(!_.contains("Small"))).flatMap(_._2).distinct).take(100).toArray
  val dimension = 10
  val imageSize = 256

  override def cudaLog = false


  override def postConfigure(log: NotebookOutput) = {
    val tableOutput = new TableOutput()
    val urlBase = "http://localhost:1080/etc/"
    NotebookRunner.withMonitoredHtml(() => tableOutput.toHtmlTable) {
      val embeddingConfigs = new ArrayBuffer[Map[String, Any]]()
      val rowCols = Math.sqrt(images.length).ceil.toInt
      val sprites = new ImgTileAssemblyLayer(rowCols, rowCols).eval(
        (images ++ images.take((rowCols * rowCols) - images.length)).par.map(VisionPipelineUtil.load(_, spriteSize, spriteSize)).map(Tensor.fromRGB(_)).toArray: _*
      ).getDataAndFree.getAndFree(0)
      val pngTxt = log.png(sprites.toRgbImage, "sprites")
      log.p(pngTxt)
      val pngFile = pngTxt.split('(').last.stripSuffix(")").stripPrefix("etc/")

      NotebookRunner.withIFrame(() => ScalaJson.toJson(Map("embeddings" -> embeddingConfigs.toArray)).toString, "json", "application/json") { fileName => {
        Try {
          val strippedname = fileName.stripPrefix("/").stripPrefix("etc/")
          val projectorUrl = s"""https://projector.tensorflow.org/?config=$urlBase$strippedname"""
          log.p(s"""<a href="$projectorUrl">Projector</a>""")
          Desktop.getDesktop.browse(new URI(projectorUrl))
        }
        for (
          styleLayer <- VGG19.values();
          op <- List(
            new ChannelMeanMatcher(),
            new GramMatrixMatcher().setTileSize(300),
            new GramMatrixEnhancer().setTileSize(300)
          )
        ) yield {
          log.h1(op.getClass.getSimpleName)
          log.h2(styleLayer.name())
          val styleVector: List[String] = log.eval(() => {
            Random.shuffle(images.toList).take(dimension)
          })
          val globalCanvas = new Tensor(imageSize, imageSize, 3)
          val trainables = (for (styleFile <- styleVector) yield {
            new VisualStyleNetwork(
              styleLayers = List(styleLayer),
              styleModifiers = List(op),
              styleUrl = List(styleFile),
              magnification = OperatorProjector.this.styleMagnification,
              minWidth = OperatorProjector.this.styleMin,
              maxWidth = OperatorProjector.this.styleMax,
              maxPixels = OperatorProjector.this.stylePixelMax,
              tileSize = 300
            ).apply(globalCanvas)
          })

          val embeddings: Map[String, Array[Double]] = (
            for (canvasFile <- images) yield {
              canvasFile -> (for ((styleFile, trainable) <- styleVector.zip(trainables).toArray) yield {
                val canvas = Tensor.fromRGB(VisionPipelineUtil.load(canvasFile, imageSize, imageSize))
                globalCanvas.set(canvas)
                val result = trainable.measure(new TrainingMonitor).sum
                canvas.freeRef()
                tableOutput.putRow(Map[CharSequence, AnyRef](
                  "source" -> styleFile,
                  "target" -> canvasFile,
                  "op_layer" -> styleLayer.name(),
                  "op_type" -> op.getClass.getSimpleName,
                  "result" -> result.asInstanceOf[java.lang.Double]
                ).asJava)
                result
              })
            }).toMap
          val id = UUID.randomUUID().toString
          log.p(log.file(images.map(embeddings).map(_.mkString("\t")).mkString("\n"), s"$id.tensors.tsv", "Data"))

          def getUrl(txt: String) = txt
            .replaceAll("s3a://simiacryptus/", "https://simiacryptus.s3.amazonaws.com/")
            .replaceAll("s3a://", "https://s3.amazonaws.com/")

          log.p(log.file((
            List(List("Label", "URL").mkString("\t")) ++
              images.map(img => List(
                img.split('/').last,
                getUrl(img)
              ).mkString("\t"))
            ).mkString("\n"), s"$id.metadata.tsv", "Metadata"))
          log.p(log.file(images.map(getUrl).mkString("\n"), s"$id.bookmarks.txt", "Bookmarks"))
          embeddingConfigs += Map(
            "tensorName" -> (op.getClass.getSimpleName + " " + styleLayer.name()),
            "tensorShape" -> Array(images.size, styleVector.size),
            "tensorPath" -> s"$urlBase$id.tensors.tsv",
            "metadataPath" -> s"$urlBase$id.metadata.tsv",
            //"bookmarksPath" -> s"$urlBase$id.bookmarks.txt",
            "sprite" -> Map(
              "imagePath" -> s"$urlBase$pngFile",
              "singleImageDim" -> Array(spriteSize, spriteSize)
            )
          )
        }
      }
      }(log)
      displayEmbedding(UUID.randomUUID().toString, urlBase, embeddingConfigs: _*)(log)
    }(log)
    null
  }

  def displayEmbedding(id: String, urlBase: String, embedding: Map[String, Any]*)(implicit log: NotebookOutput) = {
    log.p(log.file(ScalaJson.toJson(Map("embeddings" -> embedding.toArray)), s"$id.json", "Projector Config"))
    val projectorUrl = s"""https://projector.tensorflow.org/?config=$urlBase$id.json"""
    log.p(s"""<a href="$projectorUrl">Projector</a>""")
    Try {
      Desktop.getDesktop.browse(new URI(projectorUrl))
    }
    embedding
  }
}

