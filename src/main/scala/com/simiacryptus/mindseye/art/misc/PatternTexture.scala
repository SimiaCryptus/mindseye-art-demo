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

package com.simiacryptus.mindseye.art.misc

import java.lang
import java.util.concurrent.TimeUnit

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.models.VGG19._
import com.simiacryptus.mindseye.art.ops._
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util.{ArtSetup, ImageArtUtil}
import com.simiacryptus.mindseye.art.{SumTrainable, TiledTrainable, VisionPipelineLayer}
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.{ImgViewLayer, SumInputsLayer}
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.{CompoundRegion, RangeConstraint}
import com.simiacryptus.mindseye.util.ImageUtil
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

import scala.util.Random

object PatternTexture_EC2 extends PatternTexture with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object PatternTexture_Local extends PatternTexture with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5

  override def s3bucket: String = ""
}

abstract class PatternTexture extends ArtSetup[Object] {

  val styleList = Array(
    "allan-d-arcangelo",
    "rembrandt",
    "pablo-picasso",
    "david-bates",
    "henri-matisse"
    //    "jacopo-bassano",
    //    "henri-rousseau"
  )
  val inputUrl = "plasma"
  val trainingMinutes: Int = 60
  val trainingIterations: Int = 50
  val tiledViewPadding = 32
  val maxRate = 1e9
  val tileSize = 350
  val tilePadding = 32
  val styleMagnification = 1.0
  val styleMin = 64
  val styleMax = 1280
  val aspect = 1 // (11.0 - 0.5) / (8.5 - 0.5)
  val stylePixelMax = 5e6
  val patternCoeff = 1e-5
  val colorCoeff = 0
  val styleMatchCoeff = 0
  val resolutions: Array[Int] = Stream.iterate(64)(x => (x * Math.pow(1280.0 / 128, 1.0 / 8)).toInt).takeWhile(_ <= 512).toArray

  override def cudaLog = false

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> PatternTexture.this,
        "style" -> styleLayers.map(_.name())
      ))
    })
    for (styleName <- styleList) {
      log.h1(styleName)
      val patternFiles = findFiles("pikachu", base = "file:///C:/Users/andre/Downloads/pics/").take(1)
      val styleFiles = findFiles(styleName)
      var canvas: Tensor = null
      withMonitoredJpg(() => Option(canvas).map(_.toRgbImage).orNull) {
        log.subreport("pikachu", (sub: NotebookOutput) => {
          for (res <- resolutions) {
            CudaSettings.INSTANCE().defaultPrecision = precision(res)
            sub.h1("Resolution " + res)

            def loadPatterns = loadImages(
              baseImage = canvas,
              fileUrls = patternFiles,
              minWidth = 1,
              maxWidth = Integer.MAX_VALUE,
              magnification = 1.0,
              maxPixels = 5e6
            )(sub)

            def loadStyles = loadImages(
              baseImage = canvas,
              fileUrls = styleFiles,
              minWidth = styleMin,
              maxWidth = styleMax,
              magnification = styleMagnification,
              maxPixels = stylePixelMax
            )(sub)

            if (null == canvas) {
              sub.h2("Content")
              canvas = load(Array(res, (res * aspect).toInt, 3), inputUrl)(sub)
              sub.h2("Style")
              val styleImages = loadStyles
              canvas = Tensor.fromRGB(sub.eval(() => {
                colorTransfer(canvas, styleImages, false)(sub).copy().freeze().eval(canvas).getDataAndFree.getAndFree(0).toRgbImage
              }))
              sub.h2("Result")
              styleTransfer(precision(res), styleImages, loadPatterns, canvas)(sub)
            }
            else {
              canvas = Tensor.fromRGB(ImageUtil.resize(canvas.toRgbImage, res, true))
              sub.h2("Result")
              styleTransfer(
                precision = precision(res),
                styleImage = loadStyles,
                patternImages = loadPatterns,
                canvasImage = canvas
              )(sub)
            }
          }
          null
        })
      }(log)
    }
    null
  }

  def precision(w: Int) = if (w < 400) Precision.Double else Precision.Float

  def loadImages(baseImage: Tensor, fileUrls: Array[String], minWidth: Int, maxWidth: Int, magnification: Double, maxPixels: Double)(implicit log: NotebookOutput) = {
    val styles = Random.shuffle(fileUrls.toList).map(styleUrl => {
      var styleImage = ImageArtUtil.load(log, styleUrl, -1)
      val canvasDims = baseImage.getDimensions()
      val canvasPixels = canvasDims(0) * canvasDims(1)
      val stylePixels = styleImage.getWidth * styleImage.getHeight
      var finalWidth = (styleImage.getWidth * Math.sqrt((canvasPixels.toDouble / stylePixels) * magnification)).toInt
      if (finalWidth < minWidth) finalWidth = minWidth
      if (finalWidth > Math.min(maxWidth, styleImage.getWidth)) finalWidth = Math.min(maxWidth, styleImage.getWidth)
      val resized = ImageUtil.resize(styleImage, finalWidth, true)
      log.p(log.jpg(resized, styleUrl))
      Tensor.fromRGB(resized)
    }).toBuffer
    while (styles.map(_.getDimensions).map(d => d(0) * d(1)).sum > maxPixels) styles.remove(0)
    styles.foreach(style => {
      log.p(log.jpg(style.toRgbImage, ""))
    })
    styles.toArray
  }

  def styleTransfer(precision: Precision, styleImage: Seq[Tensor], patternImages: Seq[Tensor], canvasImage: Tensor)(implicit log: NotebookOutput) = {
    val styleOperator = new GramMatrixMatcher().setTileSize(tileSize).scale(styleMatchCoeff).combine(new GramMatrixEnhancer().setTileSize(tileSize).scale(styleEnhancement(canvasImage.getDimensions()(0))))
    val colorOperator = new ChannelMeanMatcher().combine(new GramMatrixMatcher().setTileSize(tileSize).scale(1e-1)).scale(colorCoeff)
    val patternOperator = new PatternPCAMatcher().scale(patternCoeff)
    val canvasDims = canvasImage.getDimensions
    val viewLayer = new ImgViewLayer(canvasDims(0) + tiledViewPadding, canvasDims(1) + tiledViewPadding, true)
      .setOffsetX(-tiledViewPadding / 2).setOffsetY(-tiledViewPadding / 2)
    val borderedPatterns = patternImages.map(patternImage => viewLayer.eval(patternImage).getDataAndFree.getAndFree(0))
    val trainable = new SumTrainable((styleLayers.groupBy(_.getPipelineName).values.toList.map(pipelineLayers => {
      val pipelineStyleLayers = pipelineLayers.filter(x => styleLayers.contains(x))
      val styleNetwork = SumInputsLayer.combine((
        patternImages.map(patternImage => colorOperator.build(patternImage)).toList ++
          pipelineStyleLayers.map(pipelineStyleLayer => styleOperator.build(pipelineStyleLayer, styleImage: _*))
          ++ pipelineStyleLayers.flatMap(pipelineStyleLayer => borderedPatterns.map(patternImage => patternOperator.build(pipelineStyleLayer, patternImage)))
        ): _*)
      //TestUtil.graph(log, styleNetwork)
      new TiledTrainable(canvasImage, viewLayer, tileSize, tilePadding, precision) {
        override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
          regionSelector.freeRef()
          MultiPrecision.setPrecision(styleNetwork.addRef(), precision).asInstanceOf[PipelineNetwork]
        }
      }
    })): _*)
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
    canvasImage
  }

  def styleEnhancement(width: Int): Double = 0 // if (width < 256) 1e1 else if (width < 512) 1e0 else 0

  def styleLayers: Seq[VisionPipelineLayer] = List(
    //    Inc5H_1a,
    //    Inc5H_2a,
    //    Inc5H_3a,
    //    Inc5H_3b,
    //    Inc5H_4a,
    //    Inc5H_4b,
    //    Inc5H_4c,
    //    Inc5H_4d,
    //    Inc5H_4e,
    //Inc5H_5a,
    //Inc5H_5b,

    //    VGG16_0,
    //    VGG16_1a,
    //    VGG16_1b1,
    //    VGG16_1b2,
    //    VGG16_1c1,
    //    VGG16_1c2,
    //    VGG16_1c3,
    //    VGG16_1d1,
    //    VGG16_1d2,
    //    VGG16_1d3
    //    VGG16_1e1,
    //    VGG16_1e2,
    //    VGG16_1e3
    //    VGG16_2

    VGG19_0b
    //    VGG19_1a,
    //    VGG19_1a2,
    //    VGG19_1b1,
    //    VGG19_1b2
    //    VGG19_1c1,
    //    VGG19_1c2,
    //    VGG19_1c3,
    //    VGG19_1c4
    //    VGG19_1d1,
    //    VGG19_1d2,
    //    VGG19_1d3,
    //    VGG19_1d4,
    //    VGG19_1e1,
    //    VGG19_1e2,
    //    VGG19_1e3,
    //    VGG19_1e4
    //    VGG19_2
  )

}
