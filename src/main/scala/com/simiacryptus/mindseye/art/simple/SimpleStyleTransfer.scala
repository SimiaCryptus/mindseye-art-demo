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

package com.simiacryptus.mindseye.art.simple

import java.lang
import java.util.concurrent.TimeUnit

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.TiledTrainable
import com.simiacryptus.mindseye.art.models.VGG16._
import com.simiacryptus.mindseye.art.ops.{ContentMatcher, GramMatrixMatcher}
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util.{ArtSetup, VisionPipelineUtil}
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.SumInputsLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.RangeConstraint
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner

object SimpleStyleTransfer_EC2 extends SimpleStyleTransfer with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object SimpleStyleTransfer_Local extends SimpleStyleTransfer with LocalRunner[Object] with NotebookRunner[Object] {
  override val contentResolution = 512
  override val styleResolution = 512

  override def inputTimeoutSeconds = 5
}

//abstract class SimpleStyleTransfer extends RepeatedArtSetup[Object] {
abstract class SimpleStyleTransfer extends ArtSetup[Object] {

  val contentUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg"
  val inputUrl = contentUrl
  val styleUrl = "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg"
  val contentResolution = 1024
  val styleResolution = 1024
  val trainingMinutes: Int = 60
  val trainingIterations: Int = 30
  val tileSize = 512
  val tilePadding = 8
  val maxRate = 1e10
  val contentCoeff = 1e-4

  override def postConfigure(log: NotebookOutput) = {
    implicit val _log = log
    CudaSettings.INSTANCE().defaultPrecision = precision
    val contentImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(contentUrl, contentResolution)
    }))
    val canvasImage = load(contentImage, inputUrl)(log)
    var styleImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(styleUrl, styleResolution)
    }))
    styleImage = colorTransfer(styleImage, List(contentImage), false)(log)
      .copy().freeze().eval(styleImage).getDataAndFree.getAndFree(0)
    val contentOperator = new ContentMatcher().scale(contentCoeff)
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
    withMonitoredJpg(() => canvasImage.toRgbImage) {
      withTrainingMonitor(trainingMonitor => {
        val trainable = new TiledTrainable(canvasImage, tileSize, tilePadding, precision) {
          override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
            val contentTile = regionSelector.eval(contentImage).getDataAndFree.getAndFree(0)
            regionSelector.freeRef()
            MultiPrecision.setPrecision(SumInputsLayer.combine(
              styleNetwork.addRef(),
              contentOperator.build(VGG16_1c1, contentTile)
            ), precision).freeze().asInstanceOf[PipelineNetwork]
          }
        }
        log.eval(() => {
          val search = new ArmijoWolfeSearch().setMaxAlpha(maxRate).setAlpha(maxRate / 10).setRelativeTolerance(1e-3)
          new IterativeTrainer(trainable)
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
      null
    }
  }

  def precision = Precision.Double

}
