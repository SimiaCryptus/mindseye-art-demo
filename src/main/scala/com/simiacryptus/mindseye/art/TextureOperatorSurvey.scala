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
import java.util.UUID
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.models.VGG19._
import com.simiacryptus.mindseye.art.ops._
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util.{ArtSetup, VisionPipelineUtil}
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.{ImgViewLayer, SumInputsLayer}
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.{CompoundRegion, RangeConstraint}
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{NotebookOutput, NullNotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

import scala.util.Random

object TextureOperatorSurvey_EC2 extends TextureOperatorSurvey with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object TextureOperatorSurvey_Local extends TextureOperatorSurvey with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5
}

class TextureOperatorSurvey extends ArtSetup[Object] {

  val inputUrl = "plasma"
  val trainingMinutes: Int = 60
  val trainingIterations: Int = 20
  val tiledViewPadding = 32
  val maxRate = 1e9
  val tileSize = 400
  val tilePadding = 32
  val aspect = 1 // (11.0 - 0.5) / (8.5 - 0.5)
  val minResolution: Double = 128
  val maxResolution: Double = 512
  val resolutionSteps: Int = 4

  val styleMagnification = 1.0
  val styleMin = 64
  val styleMax = 1280
  val stylePixelMax = 5e6

  override def cudaLog = false

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> TextureOperatorSurvey.this
      ))
    })
    for (styleKey <- List("cesar-domela", "claude-monet", "arkhip-kuindzhi")) {
      log.h1(styleKey)
      val styleFiles = findFiles(styleKey)
      loadImages(
        canvasPixels = 0,
        fileUrls = styleFiles,
        minWidth = styleMin,
        maxWidth = styleMax,
        magnification = styleMagnification,
        maxPixels = stylePixelMax
      )(log)
      for (styleLayers <- List(
        List(
          VGG19_0,
          VGG19_1a
        ), List(
          VGG19_1b1,
          VGG19_1b2
        ), List(
          VGG19_1c1,
          VGG19_1c2,
          VGG19_1c3,
          VGG19_1c4
        ), List(
          VGG19_1d1,
          VGG19_1d2,
          VGG19_1d3,
          VGG19_1d4
        ), List(
          VGG19_1e1,
          VGG19_1e2,
          VGG19_1e3,
          VGG19_1e4
        ), List(
          VGG19_0,
          VGG19_1a,
          VGG19_1b2,
          VGG19_1c4,
          VGG19_1d4,
          VGG19_1e4
        ), List(
          VGG19_0,
          VGG19_1a,
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
        )
      )) {
        log.h2(styleLayers.map(_.name()).mkString(", "))
        paintUsingLayers(styleLayers, styleFiles)(log)
      }
    }
    null
  }

  def paintUsingLayers(styleLayers: Seq[VisionPipelineLayer], styleFiles: Array[String])(implicit log: NotebookOutput) = {
    def standardTrainable(modifiers: VisualModifier*): (Tensor, Layer, Precision, Seq[Tensor]) => SumTrainable = {
      (canvas: Tensor, viewLayer: Layer, precision: Precision, styleImages: Seq[Tensor]) => {
        val styleOperator: VisualModifier = modifiers.reduce(_ combine _)
        new SumTrainable((styleLayers.groupBy(_.getPipeline.name).values.toList.map(pipelineLayers => {
          val pipelineStyleLayers = pipelineLayers.filter(x => styleLayers.contains(x))
          val styleNetwork = SumInputsLayer.combine(pipelineStyleLayers.map(styleOperator.build(_, styleImages: _*)): _*)
          new TiledTrainable(canvas, viewLayer, tileSize, tilePadding, precision) {
            override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
              regionSelector.freeRef()
              MultiPrecision.setPrecision(styleNetwork.addRef(), precision).asInstanceOf[PipelineNetwork]
            }
          }
        })): _*)
      }
    }

    paint(standardTrainable(log.eval(() => {
      List(
        new GramMatrixEnhancer().setMinMax(-0.5, 0.5).setTileSize(tileSize)
      )
    }): _*), styleFiles)
    paint(standardTrainable(log.eval(() => {
      List(
        new GramMatrixMatcher().setTileSize(tileSize)
      )
    }): _*), styleFiles)
    paint(standardTrainable(log.eval(() => {
      List(
        new ChannelMeanMatcher()
      )
    }): _*), styleFiles)
    paint(standardTrainable(log.eval(() => {
      List(
        new ChannelMeanMatcher(),
        new GramMatrixMatcher().setTileSize(tileSize)
      )
    }): _*), styleFiles)
    paint(standardTrainable(log.eval(() => {
      List(
        new GramMatrixMatcher().setTileSize(tileSize),
        new GramMatrixEnhancer().setTileSize(tileSize)
      )
    }): _*), styleFiles)
  }

  def paint(trainable: (Tensor, Layer, Precision, Seq[Tensor]) => Trainable, styleFiles: Array[String])(implicit log: NotebookOutput): Unit = {
    val canvas = new AtomicReference[Tensor](null)
    withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(canvas, trainable, styleFiles)(sub)
        null
      })
    }
  }

  def paint(canvas: AtomicReference[Tensor], trainable: (Tensor, Layer, Precision, Seq[Tensor]) => Trainable, styleFiles: Array[String])(implicit sub: NotebookOutput): Unit = {
    for (res <- resolutions) {
      val precision: Precision = this.precision(res)
      CudaSettings.INSTANCE().defaultPrecision = precision
      sub.h1("Resolution " + res)
      lazy val styleImages = loadImages(
        baseImage = canvas.get,
        fileUrls = styleFiles,
        minWidth = styleMin,
        maxWidth = styleMax,
        magnification = styleMagnification,
        maxPixels = stylePixelMax
      )(new NullNotebookOutput())
      if (null == canvas.get) {
        implicit val nullNotebookOutput = new NullNotebookOutput()
        canvas.set(load(Array(res, (res * aspect).toInt, 3), inputUrl))
        canvas.set(Tensor.fromRGB(colorTransfer(canvas.get, styleImages, false).copy().freeze().eval(canvas.get).getDataAndFree.getAndFree(0).toRgbImage))
      }
      else {
        canvas.set(Tensor.fromRGB(TestUtil.resize(canvas.get.toRgbImage, res, true)))
      }
      val canvasDims = canvas.get.getDimensions
      val viewLayer = new ImgViewLayer(canvasDims(0) + tiledViewPadding, canvasDims(1) + tiledViewPadding, true)
        .setOffsetX(-tiledViewPadding / 2).setOffsetY(-tiledViewPadding / 2)
      optimize(canvas.get, trainable(canvas.get, viewLayer, precision, styleImages))
    }
  }

  def resolutions = Stream.iterate(minResolution)(_ * growth).takeWhile(_ <= maxResolution).map(_.toInt).toArray

  private def growth = Math.pow(maxResolution / minResolution, 1.0 / resolutionSteps)

  def precision(w: Int) = if (w < 200) Precision.Double else Precision.Float

  def loadImages(baseImage: Tensor, fileUrls: Array[String], minWidth: Int, maxWidth: Int, magnification: Double, maxPixels: Double)(implicit log: NotebookOutput): Array[Tensor] = {
    val canvasDims = baseImage.getDimensions()
    val canvasPixels = canvasDims(0) * canvasDims(1)
    loadImages(
      canvasPixels = canvasPixels,
      fileUrls = fileUrls,
      minWidth = minWidth,
      maxWidth = maxWidth,
      maxPixels = maxPixels,
      magnification = magnification
    )
  }

  def optimize(canvasImage: Tensor, trainable: Trainable)(implicit log: NotebookOutput) = {
    withMonitoredJpg(canvasImage.toRgbImage) {
      withTrainingMonitor(trainingMonitor => {
        log.eval(() => {
          val search = new ArmijoWolfeSearch().setMaxAlpha(maxRate).setMinAlpha(1e-10).setAlpha(1).setRelativeTolerance(1e-5)
          IterativeTrainer.wrap(trainable)
            .setOrientation(new TrustRegionStrategy(new LBFGS) {
              override def getRegionPolicy(layer: Layer) = new CompoundRegion(
                //                new RangeConstraint().setMin(0).setMax(256),
                //                new FixedMagnitudeConstraint(canvasImage.coordStream(true)
                //                  .collect(Collectors.toList()).asScala
                //                  .groupBy(_.getCoords()(2)).values
                //                  .toArray.map(_.map(_.getIndex).toArray): _*),
                new RangeConstraint().setMin(0).setMax(256)
              )
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
  }

  def loadImages(canvasPixels: Int, fileUrls: Array[String], minWidth: Int, maxWidth: Int, maxPixels: Double, magnification: Double)(implicit log: NotebookOutput): Array[Tensor] = {
    val styles = Random.shuffle(fileUrls.toList).map(styleUrl => {
      var styleImage = VisionPipelineUtil.load(styleUrl, -1)
      val stylePixels = styleImage.getWidth * styleImage.getHeight
      var finalWidth = if (canvasPixels > 0) (styleImage.getWidth * Math.sqrt((canvasPixels.toDouble / stylePixels) * magnification)).toInt else -1
      if (finalWidth < minWidth && finalWidth > 0) finalWidth = minWidth
      if (finalWidth > Math.min(maxWidth, styleImage.getWidth)) finalWidth = Math.min(maxWidth, styleImage.getWidth)
      val resized = TestUtil.resize(styleImage, finalWidth, true)
      log.p(log.jpg(resized, styleUrl))
      Tensor.fromRGB(resized)
    }).toBuffer
    while (styles.map(_.getDimensions).map(d => d(0) * d(1)).sum > maxPixels) styles.remove(0)
    styles.foreach(style => {
      log.p(log.jpg(style.toRgbImage, ""))
    })
    styles.toArray
  }
}
