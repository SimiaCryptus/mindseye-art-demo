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

import java.awt.image.BufferedImage
import java.lang
import java.util.UUID
import java.util.concurrent.TimeUnit
import java.util.zip.{ZipEntry, ZipOutputStream}

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.models.VGG16._
import com.simiacryptus.mindseye.art.ops.{ChannelMeanMatcher, GramMatrixEnhancer, GramMatrixMatcher}
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util.{ArtSetup, ImageArtUtil}
import com.simiacryptus.mindseye.art.{SumTrainable, TiledTrainable, VisionPipelineLayer}
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Coordinate, Layer, LayerBase, Tensor}
import com.simiacryptus.mindseye.layers.ValueLayer
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
import com.simiacryptus.mindseye.layers.java._
import com.simiacryptus.mindseye.network.{DAGNode, PipelineNetwork}
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.{OrthonormalConstraint, RangeConstraint}
import com.simiacryptus.mindseye.util.ImageUtil
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner._
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}
import javax.imageio.ImageIO
import org.apache.commons.io.IOUtils

import scala.util.Random

object TileBuilder_EC2 extends TileBuilder with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object TileBuilder_Local extends TileBuilder with LocalRunner[Object] with NotebookRunner[Object] {

  override def s3bucket: String = ""

  override def inputTimeoutSeconds = 5
}

abstract class TileBuilder extends ArtSetup[Object] {

  private lazy val postPaintLayer = PipelineNetwork.wrap(1,
    new SimpleConvolutionLayer(1, 1, 3, 3).set((c: Coordinate) => {
      val coords = c.getCoords()(2)
      if ((coords % 3) == (coords / 3)) 1.0 else 0.0
    }).freeze(),
    new ImgBandBiasLayer(3).setWeights((i: Int) => 0.0).freeze()
  )

  val seed = "plasma"
  val trainingMinutes: Int = 60
  val epochIterations: Int = 5
  val trainingEpochs: Int = 3
  val maxRate = 1e10
  val tiledEvaluationPadding = 32
  val styleMagnification = 5.0
  val aspectRatio = 1.0
  val stylePixelMax = 2e6
  val tiledViewPadding = 32
  val resolutions: Array[Int] = Stream.iterate(64)(x => (x * Math.pow(2.0, 1.0 / (if (x < 128) 2 else 1))).toInt).takeWhile(_ <= 300).toArray
  val styleList = Array(
    "530116339",
    "542903440"
  )
  protected var colorCoeff = (width: Int) => 1e1

  override def cudaLog = false

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> TileBuilder.this,
        "style" -> styleLayers.map(_.name())
      ))
    })
    log.h1("Basic Textures")
    val basicTextures: Map[String, Map[Int, Tensor]] = (for (styleName <- styleList) yield {
      log.h2(styleName)
      styleName -> paint(styleName, findFiles(styleName): _*)((canvasDims: Array[Int]) => {
        new ImgViewLayer(canvasDims(0) + tiledViewPadding, canvasDims(1) + tiledViewPadding, true)
          .setOffsetX(-tiledViewPadding / 2).setOffsetY(-tiledViewPadding / 2)
      })(log)
    }).toMap
    log.eval(() => {
      ScalaJson.toJson(basicTextures.mapValues(_.mapValues(img => log.jpg(img.toRgbImage, "").split("\\(").drop(1).head.stripSuffix(")"))))
    })
    colorCoeff = (width: Int) => 1e-1
    log.h1("Intermediate Textures")
    val intermediateTextures = (for (styleName <- styleList) yield {
      log.h2(styleName)
      for (
        (topStyle, topImage) <- basicTextures;
        (bottomStyle, bottomImage) <- basicTextures;
        (leftStyle, leftImage) <- basicTextures;
        (rightStyle, rightImage) <- basicTextures
      ) yield {
        val key = Map(
          "primary" -> styleName,
          "top" -> topStyle,
          "bottom" -> bottomStyle,
          "left" -> leftStyle,
          "right" -> rightStyle
        )
        log.p(ScalaJson.toJson(key))
        key ++ Map(
          "img" -> paint(styleName, findFiles(styleName): _*)((canvasDims: Array[Int]) => {
            val bottom = ImageUtil.resize(bottomImage(canvasDims(0)).toRgbImage, canvasDims(0), true)
            val top = ImageUtil.resize(topImage(canvasDims(0)).toRgbImage, canvasDims(0), true)
            val left = ImageUtil.resize(leftImage(canvasDims(0)).toRgbImage, canvasDims(0), true)
            val right = ImageUtil.resize(rightImage(canvasDims(0)).toRgbImage, canvasDims(0), true)
            val network = new PipelineNetwork(1)
            network.wrap(new ImgTileAssemblyLayer(3, 3),
              network.wrap(new ValueLayer(selectRight(selectBottom(top, tiledViewPadding).toRgbImage, tiledViewPadding)), Array.empty[DAGNode]: _*),
              network.wrap(new ValueLayer(selectBottom(top, tiledViewPadding)), Array.empty[DAGNode]: _*),
              network.wrap(new ValueLayer(selectLeft(selectBottom(top, tiledViewPadding).toRgbImage, tiledViewPadding)), Array.empty[DAGNode]: _*),
              network.wrap(new ValueLayer(selectRight(left, tiledViewPadding)), Array.empty[DAGNode]: _*),
              network.getInput(0),
              network.wrap(new ValueLayer(selectLeft(right, tiledViewPadding)), Array.empty[DAGNode]: _*),
              network.wrap(new ValueLayer(selectRight(selectTop(bottom, tiledViewPadding).toRgbImage, tiledViewPadding)), Array.empty[DAGNode]: _*),
              network.wrap(new ValueLayer(selectTop(bottom, tiledViewPadding)), Array.empty[DAGNode]: _*),
              network.wrap(new ValueLayer(selectLeft(selectTop(bottom, tiledViewPadding).toRgbImage, tiledViewPadding)), Array.empty[DAGNode]: _*)
            )
            network
          })(log)
        )
      }
    }).flatten.toList
    log.eval(() => {
      ScalaJson.toJson(intermediateTextures.map(m => {
        m ++ Map(
          "img" -> m("img").asInstanceOf[Map[Int, Tensor]].mapValues(x => log.jpg(x.toRgbImage, "").split("\\(").drop(1).head.stripSuffix(")"))
        )
      }))
    })

    val zipName = "textures.zip"
    val zipFileOut = new ZipOutputStream(log.file(zipName))

    def getImage(img: Tensor) = {
      val id = UUID.randomUUID().toString + ".jpg"
      zipFileOut.putNextEntry(new ZipEntry(id))
      ImageIO.write(img.toRgbImage, "jpg", zipFileOut)
      zipFileOut.closeEntry()
      id
    }

    val str = ScalaJson.toJson(Map(
      "basic" -> basicTextures.mapValues(_.mapValues(getImage(_))),
      "intermediate" -> intermediateTextures.map(m => {
        m ++ Map(
          "img" -> m("img").asInstanceOf[Map[Int, Tensor]].mapValues(getImage(_))
        )
      })
    ))
    zipFileOut.putNextEntry(new ZipEntry("texture.json"))
    IOUtils.write(str, zipFileOut, "UTF-8")
    zipFileOut.closeEntry()
    zipFileOut.finish()
    zipFileOut.close()
    log.p("[%s](etc/%s)", zipName, zipName)

    null
  }

  def paint(styleName: String, styleUrl: String*)(borderLayer: Array[Int] => LayerBase)(implicit log: NotebookOutput) = {
    var canvas: Tensor = null

    def currentImage = Option(canvas).map(tensor => {
      val _borderLayer = borderLayer(tensor.getDimensions)
      val image = _borderLayer.eval(tensor).getDataAndFree.getAndFree(0).toRgbImage
      _borderLayer.freeRef()
      image
    }).orNull

    withMonitoredJpg(() => Option(canvas).map(_.toImage).orNull) {
      withMonitoredJpg(() => currentImage) {
        log.subreport[Map[Int, Tensor]](styleName, (sub: NotebookOutput) => {
          implicit val _log = sub
          (for (width <- resolutions) yield {
            CudaSettings.INSTANCE().defaultPrecision = precision(width)
            sub.h1("Resolution " + width)
            if (null == canvas) {
              canvas = load(Array(width, (width * aspectRatio).toInt), seed)()
            } else {
              canvas = Tensor.fromRGB(ImageUtil.resize(canvas.toRgbImage(), width, true))
            }
            val styleTensors = loadStyles(canvas, styleUrl: _*)(_log)
            styleTransfer(precision(width), styleTensors, styleTensors, canvas, PipelineNetwork.wrap(1,
              borderLayer(canvas.getDimensions),
              postPaintLayer.addRef(),
              new BoundedActivationLayer().setMinValue(0).setMaxValue(256)
            ))(_log)
            width -> canvas.copy()
          }).toMap
        })
      }
    }
  }

  def precision(width: Int) = if (width < 128) Precision.Double else Precision.Float

  def loadStyles(contentImage: Tensor, styleUrl: String*)(implicit log: NotebookOutput) = {
    val styles = Random.shuffle(styleUrl.toList).map(styleUrl => {
      var styleImage = ImageArtUtil.load(log, styleUrl, -1)
      val canvasDims = contentImage.getDimensions()
      val canvasPixels = canvasDims(0) * canvasDims(1)
      val stylePixels = styleImage.getWidth * styleImage.getHeight
      var finalWidth = (styleImage.getWidth * Math.sqrt((canvasPixels.toDouble / stylePixels)) * styleMagnification).toInt
      if (finalWidth > styleImage.getWidth) finalWidth = styleImage.getWidth
      val resized = ImageUtil.resize(styleImage, finalWidth, true)
      Tensor.fromRGB(resized)
    }).toBuffer
    while (styles.map(_.getDimensions).map(d => d(0) * d(1)).sum > stylePixelMax) styles.remove(0)
    styles.toArray
  }

  def styleTransfer(precision: Precision, styleImage: Seq[Tensor], colorImage: Seq[Tensor], canvasImage: Tensor, borderLayer: PipelineNetwork)(implicit log: NotebookOutput) = {
    val canvasDims = canvasImage.getDimensions()
    val currentTileSize = evaluationTileSize(precision)
    val styleOperator = new GramMatrixMatcher().setTileSize(currentTileSize)
      .combine(new GramMatrixEnhancer().setTileSize(currentTileSize).scale(styleEnhancement(canvasDims(0))))
    val colorOperator = new GramMatrixMatcher().setTileSize(currentTileSize).combine(new ChannelMeanMatcher).scale(colorCoeff(canvasDims(0)))
    val trainable = new SumTrainable((styleLayers.groupBy(_.getPipelineName).values.toList.map(pipelineLayers => {
      val pipelineStyleLayers = pipelineLayers.filter(x => styleLayers.contains(x))
      val styleNetwork = SumInputsLayer.combine((
        pipelineStyleLayers.map(pipelineStyleLayer => styleOperator.build(pipelineStyleLayer, styleImage: _*)) ++ List(
          colorOperator.build(colorImage: _*)
        )): _*)
      new TiledTrainable(canvasImage, borderLayer, currentTileSize, tiledEvaluationPadding, precision) {
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
          case layer: SimpleConvolutionLayer => new OrthonormalConstraint(ImageArtUtil.getIndexMap(layer): _*)
          case _ => null
        }
      }
      (1 to trainingEpochs).foreach(_ => {
        orientation.addRef()
        withMonitoredJpg(() => borderLayer.eval(canvasImage).getDataAndFree.getAndFree(0).toImage) {
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

  def styleEnhancement(width: Int): Double = if (width < 128) 1e1 else if (width < 200) 1e0 else 0

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

    //    VGG16_0b,
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

    //    VGG19_0b,
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

  def selectBottom(img: BufferedImage, size: Int) = {
    val positionX = 0
    val positionY = img.getHeight - size
    val width = img.getWidth
    val height = size
    selectRegion(img, positionX, positionY, width, height)
  }

  def selectTop(img: BufferedImage, size: Int) = {
    val positionX = 0
    val positionY = 0
    val width = img.getWidth
    val height = size
    selectRegion(img, positionX, positionY, width, height)
  }

  def selectRegion(img: BufferedImage, positionX: Int, positionY: Int, width: Int, height: Int) = {
    val tileSelectLayer = new ImgTileSelectLayer(width, height, positionX, positionY)
    val result = tileSelectLayer.eval(Tensor.fromRGB(img)).getDataAndFree.getAndFree(0)
    tileSelectLayer.freeRef()
    result
  }

  def selectLeft(img: BufferedImage, size: Int) = {
    val positionX = 0
    val positionY = 0
    val width = size
    val height = img.getHeight
    selectRegion(img, positionX, positionY, width, height)
  }

  def selectRight(img: BufferedImage, size: Int) = {
    val positionX = img.getTileWidth - size
    val positionY = 0
    val width = size
    val height = img.getHeight
    selectRegion(img, positionX, positionY, width, height)
  }

}

