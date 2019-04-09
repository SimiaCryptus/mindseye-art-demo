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
import com.simiacryptus.mindseye.art.constraints.{GramMatrixMatcher, RMSContentMatcher}
import com.simiacryptus.mindseye.art.models.VGG16._
import com.simiacryptus.mindseye.eval.Trainable
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
import com.simiacryptus.sparkbook.util.LocalRunner

object IteratedStyleTransfer_EC2 extends IteratedStyleTransfer with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object IteratedStyleTransfer_Local extends IteratedStyleTransfer with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5
}

//abstract class IteratedStyleTransfer extends RepeatedArtSetup[Object] {
abstract class IteratedStyleTransfer extends ArtSetup[Object] {

  //  val contentUrl = "https://s3-us-west-2.amazonaws.com/simiacryptus/photos/IMG_20181027_100612044.jpg"
  val contentUrl = "file:///C:/Users/andre/Downloads/IMG_20190329_195337154.jpg"
  val inputUrl = contentUrl
  val styleUrl = "file:///C:/Users/andre/Downloads/Paper_birds.jpg"
  val trainingMinutes: Int = 60
  val trainingIterations: Int = 15
  val maxRate = 1e10
  val tileSize = 512
  val tilePadding = 4
  val contentCoeff = 1e-4
  val styleWidthRatio = 1.5

  override def cudaLog = false

  def precision = Precision.Float

  override def postConfigure(log: NotebookOutput) = {
    implicit val _log = log
    CudaSettings.INSTANCE().defaultPrecision = precision
    var canvas: Tensor = null
    //    canvas = transfer_url(url = inputUrl, width = 1024, balanceColor = false)
    canvas = transfer_url(url = inputUrl, width = 256, balanceColor = false)
    canvas = transfer_img(img = canvas, width = 512, balanceColor = false)
    canvas = transfer_img(img = canvas, width = 1024, balanceColor = false)
    null
  }

  def transfer_url(url: String, width: Int, balanceColor: Boolean)(implicit log: NotebookOutput): Tensor = {
    val contentImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(contentUrl, width)
    }))
    val canvas = load(log, contentImage.getDimensions, url)
    var styleImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(styleUrl, (width * styleWidthRatio).toInt)
    }))
    if (balanceColor) styleImage = colorTransfer(log, styleImage, contentImage, tileSize, tilePadding, precision)
      .copy().freeze().eval(styleImage).getDataAndFree.getAndFree(0)
    stayleTransfer(log, contentCoeff, precision, contentImage, styleImage, canvas)
  }

  def transfer_img(img: Tensor, width: Int, balanceColor: Boolean)(implicit log: NotebookOutput): Tensor = {
    val contentImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(contentUrl, width)
    }))
    val canvas = Tensor.fromRGB(TestUtil.resize(img.toRgbImage, contentImage.getDimensions()(0), contentImage.getDimensions()(1)))
    var styleImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(styleUrl, (width * styleWidthRatio).toInt)
    }))
    if (balanceColor) styleImage = colorTransfer(log, styleImage, contentImage, tileSize, tilePadding, precision)
      .copy().freeze().eval(styleImage).getDataAndFree.getAndFree(0)
    stayleTransfer(log, contentCoeff, precision, contentImage, styleImage, canvas)
  }

  def stayleTransfer(log: NotebookOutput, contentCoeff: Double, precision: Precision, contentImage: Tensor, styleImage: Tensor, canvasImage: Tensor) = {
    val contentOperator = new RMSContentMatcher().scale(contentCoeff)
    val styleOperator = new GramMatrixMatcher()
    val styleNetwork: PipelineNetwork = log.eval(() => {
      SumInputsLayer.combine(
        styleOperator.build(VGG16_0, styleImage),
        styleOperator.build(VGG16_1a, styleImage),
        styleOperator.build(VGG16_1b1, styleImage),
        styleOperator.build(VGG16_1b2, styleImage),
        styleOperator.build(VGG16_1c2, styleImage),
        styleOperator.build(VGG16_1c3, styleImage)
      )
    })
    val trainable = new TiledTrainable(canvasImage, tileSize, tilePadding, precision) {
      override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
        val contentTile = regionSelector.eval(contentImage).getDataAndFree.getAndFree(0)
        regionSelector.freeRef()
        val n1 = contentOperator.build(VGG16_1c1, contentTile)
        assert(n1.currentRefCount() == 1)
        val network = MultiPrecision.setPrecision(SumInputsLayer.combine(
          styleNetwork.addRef(),
          n1
        ), precision).asInstanceOf[PipelineNetwork]
        assert(n1.isFinalized)
        contentTile.freeRef()
        network
      }
    }
    withMonitoredImage(log, canvasImage.toRgbImage) {
      train(log, trainable)
    }
    canvasImage
  }

  private def train(log: NotebookOutput, trainable: Trainable) = {
    withTrainingMonitor(log, trainingMonitor => {
      log.eval(() => {
        val search = new ArmijoWolfeSearch().setMaxAlpha(maxRate).setAlpha(maxRate / 10).setRelativeTolerance(1e-3)
        IterativeTrainer.wrap(trainable)
          .setOrientation(new TrustRegionStrategy(new LBFGS) {
            override def getRegionPolicy(layer: Layer) = new RangeConstraint().setMin(0e-2).setMax(256)
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
