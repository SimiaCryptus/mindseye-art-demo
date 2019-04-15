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

import java.lang
import java.util.concurrent.TimeUnit

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.ArtUtil._
import com.simiacryptus.mindseye.art.constraints.{GramMatrixEnhancer, GramMatrixMatcher}
import com.simiacryptus.mindseye.art.models.Inception5H._
import com.simiacryptus.mindseye.art.models.VGG16._
import com.simiacryptus.mindseye.art.models.VGG19._
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.SumInputsLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.RangeConstraint
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredImage
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

object IteratedTexture_EC2 extends IteratedTexture with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object IteratedTexture_Local extends IteratedTexture with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5
}

class IteratedTexture extends ArtSetup[Object] {

  val styleList = Array(
    "rembrandt",
    "pablo-picasso",
    "allan-d-arcangelo",
    "henri-matisse",
    "jacopo-bassano",
    "david-bates",
    "henri-rousseau"
  )
  val artBase = "s3a://simiacryptus/photos/wikiart/"
  val inputUrl = "plasma"
  val trainingMinutes: Int = 60
  val trainingIterations: Int = 50
  val maxRate = 1e10
  val tileSize = 480
  val tilePadding = 32
  val styleMagnification = 1.0
  val styleMin = 64
  val styleMax = 1280
  val aspect = 8.5 / 10
  val styleEnhancement = 1e0
  val resolutions: Array[Int] = Stream.iterate(64)(x => (x * Math.pow(2.0, 1.0 / 3)).toInt).takeWhile(_ <= 512).toArray

  override def cudaLog = false

  def precision = Precision.Float

  def styleLayers: Seq[VisionPipelineLayer] = List(
    Inc5H_1a,
    Inc5H_2a,
    Inc5H_3a,
    Inc5H_3b,
    Inc5H_4a,
    Inc5H_4b,
    Inc5H_4c,
    //Inc5H_4d,
    //Inc5H_4e,
    //Inc5H_5a,
    //Inc5H_5b,

    VGG16_0,
    VGG16_1a,
    VGG16_1b1,
    VGG16_1b2,
    VGG16_1c1,
    VGG16_1c2,
    VGG16_1c3,
    VGG16_1d1,
    VGG16_1d2,
    VGG16_1d3,
    VGG16_1e1,
    VGG16_1e2,
    VGG16_1e3,
    //    VGG16_2

    VGG19_0,
    VGG19_1a1,
    VGG19_1a2,
    VGG19_1b1,
    VGG19_1b2,
    VGG19_1c1,
    VGG19_1c2,
    VGG19_1c3,
    VGG19_1c4,
    VGG19_1d1,
    VGG19_1d2,
    VGG19_1d3,
    VGG19_1d4,
    VGG19_1e1,
    VGG19_1e2,
    VGG19_1e3,
    VGG19_1e4
    //    VGG19_2
  )

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> IteratedTexture.this,
        "style" -> styleLayers.map(_.name())
      ))
    })
    CudaSettings.INSTANCE().defaultPrecision = precision
    for (styleName <- styleList) {
      log.h1(styleName)
      val styleUrl: Array[String] = findFiles(artBase, styleName)
      var canvas: Tensor = null
      withMonitoredImage(log, () => Option(canvas).map(_.toRgbImage).orNull) {
        log.subreport(styleName, (sub: NotebookOutput) => {
          for (res <- resolutions) {
            sub.h1("Resolution " + res)
            if (null == canvas) {
              canvas = transfer_url(url = inputUrl, width = res, styleUrl)(sub)
            } else {
              canvas = transfer_img(img = canvas, width = res, styleUrl)(sub)
            }
          }
          null
        })
      }
    }
    null
  }

  def transfer_url(url: String, width: Int, styleUrl: Array[String])(implicit log: NotebookOutput): Tensor = {
    log.h2("Content")
    var canvas = load(log, Array(width, (width * aspect).toInt), url)
    log.h2("Style")
    val styleImages = loadStyles(canvas, styleUrl)
    canvas = Tensor.fromRGB(log.eval(() => {
      colorTransfer(canvas, styleImages, false)(log).copy().freeze().eval(canvas).getDataAndFree.getAndFree(0).toRgbImage
    }))
    log.h2("Result")
    stayleTransfer(precision, styleImages, canvas)
  }

  def transfer_img(img: Tensor, width: Int, styleUrl: Array[String])(implicit log: NotebookOutput): Tensor = {
    val canvas = Tensor.fromRGB(TestUtil.resize(img.toRgbImage, width, true))
    log.h2("Style")
    val styleImage = loadStyles(canvas, styleUrl)
    log.h2("Result")
    stayleTransfer(precision, styleImage, canvas)
  }

  def loadStyles(contentImage: Tensor, styleUrl: Array[String])(implicit log: NotebookOutput) = {
    styleUrl.map(styleUrl => {
      var styleImage = VisionPipelineUtil.load(styleUrl, -1)
      val canvasDims = contentImage.getDimensions()
      val canvasPixels = canvasDims(0) * canvasDims(1)
      val stylePixels = styleImage.getWidth * styleImage.getHeight
      var finalWidth = (styleImage.getWidth * Math.sqrt((canvasPixels.toDouble / stylePixels)) * styleMagnification).toInt
      if (finalWidth < styleMin) finalWidth = styleMin
      if (finalWidth > Math.min(styleMax, styleImage.getWidth)) finalWidth = Math.min(styleMax, styleImage.getWidth)
      val resized = TestUtil.resize(styleImage, finalWidth, true)
      log.p(log.jpg(resized, styleUrl))
      Tensor.fromRGB(resized)
    })
  }

  def stayleTransfer(precision: Precision, styleImage: Seq[Tensor], canvasImage: Tensor)(implicit log: NotebookOutput) = {
    val styleOperator = List(
      new GramMatrixMatcher().setTileSize(tileSize),
      new GramMatrixEnhancer().setTileSize(tileSize).scale(styleEnhancement)
    ).reduce(_.combine(_))
    val trainable = new SumTrainable((styleLayers.groupBy(_.getPipeline.name).values.toList.map(pipelineLayers => {
      val pipelineStyleLayers = pipelineLayers.filter(x => styleLayers.contains(x))
      val styleNetwork = SumInputsLayer.combine(pipelineStyleLayers.map(pipelineStyleLayer => styleOperator.build(pipelineStyleLayer, styleImage: _*)): _*)
      //TestUtil.graph(log, styleNetwork)
      new TiledTrainable(canvasImage, tileSize, tilePadding, precision) {
        override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
          regionSelector.freeRef()
          MultiPrecision.setPrecision(styleNetwork.addRef(), precision).asInstanceOf[PipelineNetwork]
        }
      }
    })): _*)
    withMonitoredImage(log, canvasImage.toRgbImage) {
      withTrainingMonitor(log, trainingMonitor => {
        log.eval(() => {
          val search = new ArmijoWolfeSearch().setMaxAlpha(maxRate).setAlpha(maxRate / 10).setRelativeTolerance(1e-3)
          IterativeTrainer.wrap(trainable)
            .setOrientation(new TrustRegionStrategy(new LBFGS) {
              override def getRegionPolicy(layer: Layer) = new RangeConstraint().setMin(0).setMax(256)
            })
            .setMonitor(trainingMonitor)
            .setTimeout(trainingMinutes, TimeUnit.MINUTES)
            .setMaxIterations(trainingIterations)
            .setLineSearchFactory((_: CharSequence) => search)
            .setTerminateThreshold(java.lang.Double.NEGATIVE_INFINITY)
            .runAndFree
            .asInstanceOf[lang.Double]
        })
      })
    }
    canvasImage
  }

}
