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

package com.simiacryptus.mindseye.art.util

import java.io.File
import java.net.{URI, URLEncoder}
import java.text.Normalizer
import java.util
import java.util.concurrent.atomic.AtomicReference

import com.amazonaws.services.ec2.{AmazonEC2, AmazonEC2ClientBuilder}
import com.amazonaws.services.s3.{AmazonS3, AmazonS3ClientBuilder}
import com.fasterxml.jackson.annotation.JsonIgnore
import com.google.gson.GsonBuilder
import com.simiacryptus.aws.{EC2Util, S3Util}
import com.simiacryptus.mindseye.art.registry.{GifRegistration, JpgRegistration}
import com.simiacryptus.mindseye.art.util.ArtUtil.{cyclicalAnimation, load}
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput, NullNotebookOutput}
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.{InteractiveSetup, NotebookRunner, RepeatedInteractiveSetup}
import com.simiacryptus.util.FastRandom
import org.apache.commons.io.{FileUtils, IOUtils}

import scala.collection.JavaConversions._
import scala.concurrent.{ExecutionContext, Future}

object ArtSetup {
  @JsonIgnore
  @transient implicit val s3client: AmazonS3 = AmazonS3ClientBuilder.standard().withRegion(EC2Util.REGION).build()
  @JsonIgnore
  @transient implicit val ec2client: AmazonEC2 = AmazonEC2ClientBuilder.standard().withRegion(EC2Util.REGION).build()

}

import com.simiacryptus.mindseye.art.util.ArtSetup._

trait ArtSetup[T <: AnyRef] extends InteractiveSetup[T] {

  val label = "Demo"

  def s3bucket: String

  def registerWithIndex(canvas: Seq[AtomicReference[Tensor]])(implicit log: NotebookOutput) = {
    val archiveHome = log.getArchiveHome
    if (!s3bucket.isEmpty && null != archiveHome) Option(new GifRegistration(
      bucket = s3bucket.split("/").head,
      reportUrl = "http://" + archiveHome.getHost + "/" + archiveHome.getPath.stripSuffix("/").stripPrefix("/") + "/" + log.getName + ".html",
      liveUrl = s"http://${EC2Util.publicHostname()}:1080/",
      canvas = () => {
        var list = canvas.map(_.get()).filter(_ != null).map(_.toImage)
        val maxWidth = list.map(_.getWidth).max
        list = list.map(TestUtil.resize(_, maxWidth, true))
        (list ++ list.reverse.tail.dropRight(1))
      }
    ).start()(s3client, ec2client)) else None
  }

  def registerWithIndex(canvas: AtomicReference[Tensor])(implicit log: NotebookOutput) = {
    val archiveHome = log.getArchiveHome
    if (!s3bucket.isEmpty && null != archiveHome) Option(new JpgRegistration(
      bucket = s3bucket.split("/").head,
      reportUrl = "http://" + archiveHome.getHost + "/" + archiveHome.getPath.stripSuffix("/").stripPrefix("/") + "/" + log.getName + ".html",
      liveUrl = s"http://${EC2Util.publicHostname()}:1080/",
      canvas = () => canvas.get()
    ).start()(s3client, ec2client)) else None
  }

  def upload(log: NotebookOutput)(implicit executionContext: ExecutionContext = ExecutionContext.global) = {
    log.write()
    for (home <- Option(log.getArchiveHome).filter(!_.toString.isEmpty)) {
      Future {
        S3Util.upload(s3client, home, log.getRoot)
      }
    }
  }

  def getPaintingsBySearch(searchWord: String, minWidth: Int): Array[String] = {
    getPaintings(new URI("https://www.wikiart.org/en/search/" + URLEncoder.encode(searchWord, "UTF-8").replaceAllLiterally("+", "%20") + "/1?json=2"), minWidth, 100)
  }

  def getPaintingsByArtist(artist: String, minWidth: Int): Array[String] = {
    getPaintings(new URI("https://www.wikiart.org/en/App/Painting/PaintingsByArtist?artistUrl=" + artist), minWidth, 100)
  }

  def getPaintings(uri: URI, minWidth: Int, maxResults: Int): Array[String] = {
    new GsonBuilder().create().fromJson(IOUtils.toString(
      uri,
      "UTF-8"
    ), classOf[util.ArrayList[util.Map[String, AnyRef]]])
      .filter(_ ("width").asInstanceOf[Number].doubleValue() > minWidth)
      .map(_ ("image").toString.stripSuffix("!Large.jpg"))
      .take(maxResults)
      .map(file => {
        val fileName = Normalizer.normalize(
          file.split("/").takeRight(2).mkString("/"),
          Normalizer.Form.NFD
        ).replaceAll("[^\\p{ASCII}]", "")
        val localFile = new File(new File("wikiart"), fileName)
        try {
          if (!localFile.exists()) {
            FileUtils.writeByteArrayToFile(localFile, IOUtils.toByteArray(new URI(file)))
          }
          "file:///" + localFile.getAbsolutePath.replaceAllLiterally("\\", "/").stripPrefix("/")
        } catch {
          case e: Throwable =>
            e.printStackTrace()
            ""
        }
      }).filterNot(_.isEmpty).toArray
  }

  def paintBisection(contentUrl: String, initUrl: String, canvases: Seq[AtomicReference[Tensor]], networks: Seq[(String, VisualNetwork)], optimizer: BasicOptimizer, renderingFn: Seq[Int] => PipelineNetwork, chunks: Int, resolutions: Double*)(implicit sub: NotebookOutput) = {
    paint(contentUrl, initUrl, canvases, networks, optimizer, resolutions, toBisectionChunks(0 until networks.size, chunks), renderingFn)
  }

  def toBisectionChunks(seq: Seq[Int], chunks: Int = 1): Seq[Int] = {
    def middleFirst(seq: Seq[Int]): Seq[Int] = {
      if (seq.isEmpty) seq
      else {
        val leftNum = (seq.size.toDouble / 2).floor.toInt
        val rightNum = (seq.size.toDouble / 2).ceil.toInt - 1
        seq.drop(leftNum).dropRight(rightNum) ++ middleFirst(seq.take(leftNum)) ++ middleFirst(seq.takeRight(rightNum))
      }
    }

    if (seq.size < 3) seq
    else {
      val thisChunk = ((seq.size - (chunks + 1)) / chunks).ceil.toInt
      seq.take(1) ++ toBisectionChunks(seq.drop(1 + thisChunk), chunks - 1) ++ middleFirst(seq.drop(1).take(thisChunk))
    }
  }

  def paintOrdered(contentUrl: String, initUrl: String, canvases: Seq[AtomicReference[Tensor]], networks: Seq[(String, VisualNetwork)], optimizer: BasicOptimizer, renderingFn: Seq[Int] => PipelineNetwork, resolutions: Double*)(implicit sub: NotebookOutput) = {
    paint(contentUrl, initUrl, canvases, networks, optimizer, resolutions, (0 until networks.size), renderingFn)
  }

  def paint(contentUrl: String, initUrl: String, canvases: Seq[AtomicReference[Tensor]], networks: Seq[(String, VisualNetwork)], optimizer: BasicOptimizer, resolutions: Seq[Double], seq: Seq[Int], renderingFn: Seq[Int] => PipelineNetwork)(implicit sub: NotebookOutput) = {
    for (res <- resolutions) {
      sub.h1("Resolution " + res)
      NotebookRunner.withMonitoredGif(() => {
        cyclicalAnimation(canvases.map(_.get()).filter(_ != null).map(tensor => {
          renderingFn(tensor.getDimensions).eval(tensor).getDataAndFree.getAndFree(0)
        }))
      }) {
        for (i <- seq) {
          val (name, network) = networks(i)
          sub.h2(name)
          val canvas = canvases(i)
          CudaSettings.INSTANCE().defaultPrecision = network.precision
          val content = VisionPipelineUtil.load(contentUrl, res.toInt)
          if (null == canvas.get) {
            implicit val nullNotebookOutput = new NullNotebookOutput()
            val l = canvases.zipWithIndex.take(i).filter(_._1.get() != null).lastOption
            val r = canvases.zipWithIndex.drop(i + 1).reverse.filter(_._1.get() != null).lastOption
            if (l.isDefined && r.isDefined && l.get._2 != r.get._2) {
              canvas.set(l.get._1.get().add(r.get._1.get()).scaleInPlace(0.5))
            } else {
              canvas.set(load(Tensor.fromRGB(content), initUrl))
            }
          }
          else {
            canvas.set(Tensor.fromRGB(TestUtil.resize(canvas.get.toRgbImage, content.getWidth, content.getHeight)))
          }
          val trainable = network(canvas.get, Tensor.fromRGB(content))
          ArtUtil.resetPrecision(trainable, network.precision)
          optimizer.optimize(canvas.get, trainable)
        }
      }
    }
  }

  def paint(contentUrl: String, initUrl: String, canvas: AtomicReference[Tensor], network: VisualNetwork, optimizer: BasicOptimizer, resolutions: Double*)(implicit sub: NotebookOutput): Unit = {
    for (res <- resolutions) {
      CudaSettings.INSTANCE().defaultPrecision = network.precision
      sub.h1("Resolution " + res)
      var content = VisionPipelineUtil.load(contentUrl, res.toInt)
      val contentTensor = if (null == content) {
        new Tensor(res.toInt, res.toInt, 3).mapAndFree((x: Double) => FastRandom.INSTANCE.random())
      } else {
        Tensor.fromRGB(content)
      }
      if (null == content) content = contentTensor.toImage
      require(null != canvas)

      def updateCanvas(currentCanvas: Tensor) = {
        if (null == currentCanvas) {
          load(contentTensor, initUrl)
        } else {
          val width = if (null == content) res.toInt else content.getWidth
          val height = if (null == content) res.toInt else content.getHeight
          Tensor.fromRGB(TestUtil.resize(currentCanvas.toRgbImage, width, height))
        }
      }

      val currentCanvas: Tensor = updateCanvas(canvas.get())
      canvas.set(currentCanvas)
      val trainable = network.apply(currentCanvas, contentTensor)
      ArtUtil.resetPrecision(trainable, network.precision)
      optimizer.optimize(currentCanvas, trainable)
    }
  }

  override def apply(log: NotebookOutput): T = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    VisionPipelineUtil.cudaReports(log, cudaLog)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(maxImageSize)
    super.apply(log)
  }

  def cudaLog = false

  def maxImageSize = 10000
}

trait RepeatedArtSetup[T <: AnyRef] extends RepeatedInteractiveSetup[T] {
  val label = "Demo"

  override def apply(log: NotebookOutput): T = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    VisionPipelineUtil.cudaReports(log, cudaLog)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(maxImageSize)
    super.apply(log)
  }

  def cudaLog = false

  def maxImageSize = 10000
}
