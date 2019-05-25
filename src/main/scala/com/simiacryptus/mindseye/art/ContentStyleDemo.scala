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

import java.util.UUID
import java.util.concurrent.atomic.AtomicReference

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.models.VGG19._
import com.simiacryptus.mindseye.art.models.PoolingPipeline._
import com.simiacryptus.mindseye.art.models.VGG19
import com.simiacryptus.mindseye.art.ops.{ChannelMeanMatcher, GramMatrixEnhancer, GramMatrixMatcher, ContentMatcher}
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util._
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, Precision}
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{NotebookOutput, NullNotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

object ContentStyleDemo_EC2 extends ContentStyleDemo with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object ContentStyleDemo_Local extends ContentStyleDemo with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5
}

class ContentStyleDemo extends ArtSetup[Object] with BasicOptimizer with GeometricResolutionSequence {
  override val minResolution = 400
  override val maxResolution = 4000
  override val resolutionSteps = 10
  override val trainingMinutes: Int = 90
  override val trainingIterations: Int = 50
  override val maxRate = 1e9
//  val contentUrl = "file:///C:/Users/andre/Downloads/IMG_20190422_150855449.jpg"
  val contentUrl = "file:///E:/Media/Pictures/moms_photoframe/moms_photoframe-0043.jpg"
  val styleUrls = Array(
    "https://uploads4.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg"
  )
  val initUrl = "noise100"

  override def cudaLog = false

  def precision(w: Int) = if (w < 200) Precision.Double else Precision.Float

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> ContentStyleDemo.this
      ))
    })
    val styleNetwork = new CartesianStyleNetwork(
      styleLayers = List(
        VGG19.VGG19_1b2,
        VGG19.VGG19_1c1,
        VGG19.VGG19_1c2,
        VGG19.VGG19_1c3,
        VGG19.VGG19_1c4,
        VGG19.VGG19_1d2,
        VGG19.VGG19_1d3,
        VGG19.VGG19_1d4
      ),
      styleModifiers = List(
        new ChannelMeanMatcher(),
        new GramMatrixMatcher().setTileSize(400),
        new GramMatrixEnhancer().setMinMax(-0.25, 0.25).setTileSize(400)
      ),
      styleUrl = styleUrls
    ).withContent(List(
      Pooling16,
      VGG19_1d1
    ), List(
      new ContentMatcher().scale(1e3)
    )).copy(
      magnification = 1.0,
      maxWidth = 4000,
      maxPixels = 1e8,
      tileSize = 512,
      tilePadding = 8,
      precision = Precision.Float
    )

    val canvas = new AtomicReference[Tensor](null)
    withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(canvas, styleNetwork)(sub)
        null
      })
      null
    } (log)

    null
  }

  def paint(canvas: AtomicReference[Tensor], network: CartesianStyleContentNetwork)(implicit sub: NotebookOutput): Unit = {
    for (res <- resolutions) {
      val precision: Precision = this.precision(res)
      CudaSettings.INSTANCE().defaultPrecision = precision
      sub.h1("Resolution " + res)
      val content = VisionPipelineUtil.load(contentUrl, res)
      if (null == canvas.get) {
        implicit val nullNotebookOutput = new NullNotebookOutput()
        canvas.set(load(Tensor.fromRGB(content), initUrl))
      }
      else {
        canvas.set(Tensor.fromRGB(TestUtil.resize(canvas.get.toRgbImage, content.getWidth, content.getHeight)))
      }
      optimize(canvas.get, network.copy(precision = precision).apply(canvas.get, Tensor.fromRGB(content)))
    }
  }

}


