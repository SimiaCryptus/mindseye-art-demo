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

import java.awt.Font
import java.awt.image.BufferedImage
import java.lang
import java.util.concurrent.TimeUnit

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.models.VGG16._
import com.simiacryptus.mindseye.art.ops.{GramMatrixEnhancer, GramMatrixMatcher}
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util.{ArtSetup, TextUtil, VisionPipelineUtil}
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Coordinate, Layer, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
import com.simiacryptus.mindseye.layers.java._
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.{OrthonormalConstraint, RangeConstraint}
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner._
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

import scala.util.Random

object Stereogram_EC2 extends Stereogram with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object Stereogram_Local extends Stereogram with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5
}

class Stereogram extends ArtSetup[Object] {

  val colorImg = "piazza-d-italia"
  val styleList = Array(
    "guido-reni",
    "henri-matisse",
    "pablo-picasso",
    "allan-d-arcangelo",
    "claude-monet",
    "giorgio-de-chirico",
    "david-bates",
    "cornelis-springer",
    "rembrandt",
    "jacopo-bassano",
    "henri-rousseau"
  )
  val seed = "plasma"
  val trainingMinutes: Int = 30
  val epochIterations: Int = 5
  val trainingEpochs: Int = 5
  val maxRate = 1e6
  val tiledEvaluationPadding = 32
  val styleMagnification = 8.0
  val aspectRatio = 6.0
  val stylePixelMax = 2e6
  val tiledViewPadding = 32
  val resolutions: Array[Int] = Array(100)
  val depthFactor = 8
  val posterWidth = 1400
  val text = "PLATO"

  override def cudaLog = false

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> Stereogram.this,
        "style" -> styleLayers.map(_.name())
      ))
    })
    //    import scala.concurrent.ExecutionContext.Implicits.global
    //    log.subreport("Fonts", (subreport: NotebookOutput) => Future {
    //      GraphicsEnvironment.getLocalGraphicsEnvironment.getAvailableFontFamilyNames.filter((x: String) => !(x == "EmojiOne Color")).foreach((fontname: String) => {
    //        subreport.p(fontname)
    //        subreport.p(subreport.png(TextUtil.draw(text, 800, 20, fontname, Font.PLAIN), fontname))
    //        subreport.p(subreport.png(TextUtil.draw(text, 800, 20, fontname, Font.ITALIC), fontname))
    //        subreport.p(subreport.png(TextUtil.draw(text, 800, 20, fontname, Font.BOLD), fontname))
    //      })
    //      null
    //    })
    for (styleName <- styleList) {
      log.h1(styleName)
      val styleUrl: Array[String] = findFiles(styleName)
      val colorUrl = findFiles(colorImg).head
      var canvas: Tensor = null

      def currentImage = Option(canvas).map(_.toRgbImage).orNull

      lazy val depthImg = depthMap(text)
      withMonitoredJpg(() => stereoImage(depthMap = depthImg, canvas = canvas, depthFactor)) {
        log.subreport(styleName, (sub: NotebookOutput) => {
          implicit val _log = sub
          for (width <- resolutions) {
            CudaSettings.INSTANCE().defaultPrecision = precision(width)
            sub.h1("Resolution " + width)
            if (null == canvas) {
              canvas = load(Array(width, (width * aspectRatio).toInt), seed)()
              //canvas = Tensor.fromRGB(colorTransfer(canvas, styleImages, false).eval(canvas).getDataAndFree.getAndFree(0).toRgbImage)
            } else {
              canvas = Tensor.fromRGB(TestUtil.resize(canvas.toRgbImage(), width, true))
            }
            val styleTensors = loadStyles(canvas, styleUrl: _*)
            val colorTensors = loadStyles(canvas, colorUrl)
            styleTransfer(precision(width), styleTensors, colorTensors, canvas)
          }
          null
        })
        null
      } {
        log
      }
    }
    null
  }

  def precision(width: Int) = if (width < 0) Precision.Double else Precision.Float

  def depthMap(text: String, fontName: String = "Calibri"): Tensor = Tensor.fromRGB(TextUtil.draw(text, posterWidth, 120, fontName, Font.BOLD | Font.CENTER_BASELINE))

  def stereoImage(depthMap: Tensor, canvas: Tensor, depthFactor: Int): BufferedImage = {
    val dimensions = canvas.getDimensions
    val canvasWidth = dimensions(0)
    val depthScale = canvasWidth / (depthFactor * depthMap.getData.max)

    def getPixel(x: Int, y: Int, c: Int): Double = {
      if (x < 0) getPixel(x + canvasWidth, y, c)
      else if (x < canvasWidth) canvas.get(x, y % dimensions(1), c)
      else {
        val depth = depthMap.get(x, y, c)
        if (0 == depth) canvas.get(x % canvasWidth, y % dimensions(1), c)
        else getPixel(x - canvasWidth + (depthScale * depth).toInt, y, c)
      }
    }

    depthMap.copy().setByCoord((c: Coordinate) => {
      val ints = c.getCoords()
      getPixel(ints(0), ints(1), ints(2))
    }, true).toRgbImage
  }

  def loadStyles(contentImage: Tensor, styleUrl: String*) = {
    val styles = Random.shuffle(styleUrl.toList).map(styleUrl => {
      var styleImage = VisionPipelineUtil.load(styleUrl, -1)
      val canvasDims = contentImage.getDimensions()
      val canvasPixels = canvasDims(0) * canvasDims(1)
      val stylePixels = styleImage.getWidth * styleImage.getHeight
      var finalWidth = (styleImage.getWidth * Math.sqrt((canvasPixels.toDouble / stylePixels)) * styleMagnification).toInt
      if (finalWidth > styleImage.getWidth) finalWidth = styleImage.getWidth
      val resized = TestUtil.resize(styleImage, finalWidth, true)
      Tensor.fromRGB(resized)
    }).toBuffer
    while (styles.map(_.getDimensions).map(d => d(0) * d(1)).sum > stylePixelMax) styles.remove(0)
    styles.toArray
  }

  def styleTransfer(precision: Precision, styleImage: Seq[Tensor], colorImage: Seq[Tensor], canvasImage: Tensor)(implicit log: NotebookOutput) = {
    val canvasDims = canvasImage.getDimensions()
    val viewLayer = PipelineNetwork.wrap(1,
      new ImgViewLayer(canvasDims(0) + tiledViewPadding, canvasDims(1) + tiledViewPadding, true)
        .setOffsetX(-tiledViewPadding / 2).setOffsetY(-tiledViewPadding / 2)
    )
    val currentTileSize = evaluationTileSize(precision)
    val styleOperator = new GramMatrixMatcher().setTileSize(currentTileSize)
      .combine(new GramMatrixEnhancer().setTileSize(currentTileSize).scale(styleEnhancement(canvasDims(0))))
    val colorOperator = new GramMatrixMatcher().setTileSize(currentTileSize).scale(colorCoeff(canvasDims(0)))
    val trainable = new SumTrainable((styleLayers.groupBy(_.getPipeline.name).values.toList.map(pipelineLayers => {
      val pipelineStyleLayers = pipelineLayers.filter(x => styleLayers.contains(x))
      val styleNetwork = SumInputsLayer.combine((
        pipelineStyleLayers.map(pipelineStyleLayer => styleOperator.build(pipelineStyleLayer, styleImage: _*)) ++ List(
          colorOperator.build(colorImage: _*)
        )): _*)
      new TiledTrainable(canvasImage, viewLayer, currentTileSize, tiledEvaluationPadding, precision) {
        override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
          regionSelector.freeRef()
          MultiPrecision.setPrecision(styleNetwork.addRef(), precision).asInstanceOf[PipelineNetwork]
        }
      }
    })): _*)
    withTrainingMonitor(trainingMonitor => {
      val search = new ArmijoWolfeSearch().setMaxAlpha(maxRate).setAlpha(maxRate / 10).setRelativeTolerance(1e-3)
      val orientation = new TrustRegionStrategy(new LBFGS) {
        override def getRegionPolicy(layer: Layer) = layer match {
          case null => new RangeConstraint().setMin(0).setMax(256)
          case layer if layer.isFrozen => null
          case layer: SimpleConvolutionLayer => new OrthonormalConstraint(VisionPipelineUtil.getIndexMap(layer): _*)
          case _ => null
        }
      }
      (1 to trainingEpochs).foreach(_ => {
        orientation.addRef()
        withMonitoredJpg(() => canvasImage.toImage) {
          new IterativeTrainer(trainable)
            .setOrientation(orientation)
            .setMonitor(trainingMonitor)
            .setTimeout(trainingMinutes, TimeUnit.MINUTES)
            .setMaxIterations(epochIterations)
            .setLineSearchFactory((_: CharSequence) => search)
            .setTerminateThreshold(java.lang.Double.NEGATIVE_INFINITY)
            .runAndFree
            .asInstanceOf[lang.Double]
        }
      })
    })
    canvasImage
  }

  def colorCoeff(width: Int) = 1e1

  def styleEnhancement(width: Int): Double = 1e1

  def evaluationTileSize(precision: Precision) = if (precision == Precision.Double) 256 else 400

  def styleLayers: Seq[VisionPipelineLayer] = List(
    //    Inc5H_1a,
    //    Inc5H_2a,
    //    Inc5H_3a,
    //    Inc5H_3b,
    //    Inc5H_4a,
    //    Inc5H_4b,
    //    Inc5H_4c,
    //Inc5H_4d,
    //Inc5H_4e,
    //Inc5H_5a,
    //Inc5H_5b,

    //    VGG16_0,
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
    VGG16_1e3
    //    VGG16_2

    //    VGG19_0,
    //    VGG19_1a,
    //    VGG19_1a2,
    //    VGG19_1b1,
    //    VGG19_1b2,
    //    VGG19_1c1,
    //    VGG19_1c2,
    //    VGG19_1c3,
    //    VGG19_1c4,
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

