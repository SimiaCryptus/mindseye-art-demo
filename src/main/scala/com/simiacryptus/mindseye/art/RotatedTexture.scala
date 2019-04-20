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
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.constraints.{GramMatrixEnhancer, GramMatrixMatcher}
import com.simiacryptus.mindseye.art.models.VGG16._
import com.simiacryptus.mindseye.art.util.{ArtSetup, Permutation}
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Coordinate, Layer, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
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

object RotatedTexture_EC2 extends RotatedTexture with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object RotatedTexture_Local extends RotatedTexture with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5
}

class RotatedTexture extends ArtSetup[Object] {

  private lazy val postPaintLayer = PipelineNetwork.wrap(1,
    new SimpleConvolutionLayer(1, 1, 3, 3).set((c: Coordinate) => {
      val coords = c.getCoords()(2)
      if ((coords % 3) == (coords / 3)) 1.0 else 0.0
    }).freeze(),
    new ImgBandBiasLayer(3).setWeights((i: Int) => 0.0).freeze()
  )
  val colorImg = "piazza-d-italia"
  val styleList = Array(
    "guido-reni",
    "henri-matisse",
    "pablo-picasso",
    "allan-d-arcangelo",
    "claude-monet",
    "giorgio-de-chirico",
    "david-bates",
    "cornelis-springer"
    //    "rembrandt",
    //    "jacopo-bassano",
    //    "henri-rousseau"
  )
  val seed = "plasma"
  val trainingMinutes: Int = 60
  val epochIterations: Int = 5
  val trainingEpochs: Int = 5
  val maxRate = 1e10
  val tiledEvaluationPadding = 32
  val styleMagnification = 1.0
  val aspectRatio = 1.0 // (0.5+Math.sqrt(5.0/4.0))
  val stylePixelMax = 2e6
  val rotationalSegments = 2
  val rotationalChannelPermutation = Permutation.roots(3, rotationalSegments).head.indices
  val tiledViewPadding = 32
  val resolutions: Array[Int] = Stream.iterate(64)(x => (x * Math.pow(2.0, 1.0 / (if (x < 512) 3 else 2))).toInt).takeWhile(_ <= 800).toArray

  override def cudaLog = false

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> RotatedTexture.this,
        "style" -> styleLayers.map(_.name())
      ))
    })
    for (styleName <- styleList) {
      log.h1(styleName)
      val styleUrl: Array[String] = findFiles(styleName)
      val colorUrl = findFiles(colorImg).head
      var canvas: Tensor = null

      def currentImage = Option(canvas).map(tensor => {
        val kaleidoscope = getKaleidoscope(tensor.getDimensions)
        val image = kaleidoscope.eval(tensor).getDataAndFree.getAndFree(0).toRgbImage
        kaleidoscope.freeRef()
        image
      }).orNull

      def currentAnimation = {
        val image = Tensor.fromRGB(currentImage)
        val arc = 2 * Math.PI / rotationalSegments
        val permutation = Permutation(this.rotationalChannelPermutation: _*).matrix
        val identity = Permutation.unity(3).matrix
        val frames = 16
        (0.0 until 1.0 by 1.0 / frames).map(time => {
          val rotor = getRotor(arc * time, image.getDimensions)
          val sin = Math.sin(0.5 * Math.PI * time)
          val cos = Math.cos(0.5 * Math.PI * time)
          val root = permutation.scalarMultiply(sin).add(identity.scalarMultiply(cos))
          val paletteBias = new ImgBandBiasLayer(3)
          for (i <- (0 until 3)) {
            if (rotationalChannelPermutation(i) < 0) paletteBias.getBias()(i) = 256 * sin
          }
          val palette = new SimpleConvolutionLayer(1, 1, 3, 3)
          palette.kernel.setByCoord((c: Coordinate) => {
            val x = c.getCoords()(2)
            root.getEntry(x % 3, x / 3)
          })
          val frameView = PipelineNetwork.wrap(1, rotor, palette, paletteBias)
          try {
            frameView.eval(image).getDataAndFree.getAndFree(0).toImage
          } finally {
            frameView.freeRef()
          }
        })
      }

      withMonitoredJpg(() => currentImage) {
        withMonitoredGif(() => currentAnimation) {
          withMonitoredJpg(() => imageGrid(currentImage)) {
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
                val colorTensors = loadStyles(canvas, colorUrl).map(tensor => {
                  val kaleidoscope = getKaleidoscope(tensor.getDimensions)
                  val image = kaleidoscope.eval(tensor).getDataAndFree.getAndFree(0)
                  kaleidoscope.freeRef()
                  image
                })
                styleTransfer(precision(width), styleTensors, colorTensors, canvas)
              }
              null
            })
            null
          } {
            log
          }
          null
        } {
          log
        }
        null
      } {
        log
      }
    }
    null
  }

  def precision(width: Int) = if (width < 300) Precision.Double else Precision.Float

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
    val kaleidoscopeLayer = PipelineNetwork.wrap(1,
      getKaleidoscope(canvasDims),
      postPaintLayer.addRef(),
      new BoundedActivationLayer().setMinValue(0).setMaxValue(256)
    )
    val viewLayer = PipelineNetwork.wrap(1,
      kaleidoscopeLayer.addRef(),
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
        withMonitoredJpg(() => kaleidoscopeLayer.eval(canvasImage).getDataAndFree.getAndFree(0).toImage) {
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

  def styleEnhancement(width: Int): Double = if (width < 300) 1e1 else if (width < 600) 1e0 else 0

  def evaluationTileSize(precision: Precision) = if (precision == Precision.Double) 256 else 512

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
    //    VGG16_1a,
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
    //    VGG19_1a1,
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

  def getKaleidoscope(canvasDims: Array[Int]) = {
    val permutation = Permutation(this.rotationalChannelPermutation: _*)
    require(permutation.unity == (permutation ^ rotationalSegments), s"$permutation ^ $rotationalSegments => ${(permutation ^ rotationalSegments)} != ${permutation.unity}")
    val network = new PipelineNetwork(1)
    network.add(new SumInputsLayer(), (0 until rotationalSegments)
      .map(segment => {
        if (0 == segment) network.getInput(0) else {
          network.wrap(
            getRotor(segment * 2 * Math.PI / rotationalSegments, canvasDims).setChannelSelector((permutation ^ segment).indices: _*),
            network.getInput(0)
          )
        }
      }): _*).freeRef()
    network.wrap(new LinearActivationLayer().setScale(1.0 / rotationalSegments).freeze()).freeRef()
    network
  }

  def getRotor(radians: Double, canvasDims: Array[Int]) = {
    new ImgViewLayer(canvasDims(0), canvasDims(1), true)
      .setRotationCenterX(canvasDims(0) / 2)
      .setRotationCenterY(canvasDims(1) / 2)
      .setRotationRadians(radians)
  }
}

