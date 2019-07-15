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

import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.google.gson.GsonBuilder
import com.simiacryptus.aws.{EC2Util, S3Util}
import com.simiacryptus.mindseye.art.util.ArtUtil.load
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput, NullNotebookOutput}
import com.simiacryptus.sparkbook.{InteractiveSetup, RepeatedInteractiveSetup}
import org.apache.commons.io.{FileUtils, IOUtils}

import scala.collection.JavaConversions._

object ArtSetup {
  lazy val s3client = AmazonS3ClientBuilder.standard.withRegion(EC2Util.REGION).build

}

import com.simiacryptus.mindseye.art.util.ArtSetup._

trait ArtSetup[T <: AnyRef] extends InteractiveSetup[T] {
  val label = "Demo"

  def upload(log: NotebookOutput) = {
    S3Util.upload(s3client, log.getArchiveHome, log.getRoot)
  }

  def getPaintingsBySearch(searchWord: String, minWidth: Int): Array[String] = {
    getPaintings(new URI("https://www.wikiart.org/en/search/" + URLEncoder.encode(searchWord, "UTF-8").replaceAllLiterally("+", "%20") + "/1?json=2"), minWidth, 100)
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

  def getPaintingsByArtist(artist: String, minWidth: Int): Array[String] = {
    getPaintings(new URI("https://www.wikiart.org/en/App/Painting/PaintingsByArtist?artistUrl=" + artist), minWidth, 100)
  }

  def paint(contentUrl: String, initUrl: String, canvas: AtomicReference[Tensor], network: CartesianNetwork, optimizer: BasicOptimizer, resolutions: Int*)(implicit sub: NotebookOutput): Unit = {
    for (res <- resolutions) {
      CudaSettings.INSTANCE().defaultPrecision = network.precision
      sub.h1("Resolution " + res)
      val content = VisionPipelineUtil.load(contentUrl, res)
      if (null == canvas.get) {
        implicit val nullNotebookOutput = new NullNotebookOutput()
        canvas.set(load(Tensor.fromRGB(content), initUrl))
      }
      else {
        canvas.set(Tensor.fromRGB(TestUtil.resize(canvas.get.toRgbImage, content.getWidth, content.getHeight)))
      }
      val trainable = network.apply(canvas.get, Tensor.fromRGB(content))
      ArtUtil.resetPrecision(trainable, network.precision)
      optimizer.optimize(canvas.get, trainable)
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
