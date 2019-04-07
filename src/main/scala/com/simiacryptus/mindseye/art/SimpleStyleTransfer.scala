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
import com.simiacryptus.mindseye.art.constraints.{ChannelMeanMatcher, GramMatrixMatcher, RMSContentMatcher}
import com.simiacryptus.mindseye.art.models.Inception5H._
import com.simiacryptus.mindseye.lang.cudnn.{CudaMemory, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.SumInputsLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.BisectionSearch
import com.simiacryptus.mindseye.opt.orient.{GradientDescent, LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.RangeConstraint
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredImage
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner

object SimpleStyleTransfer_EC2 extends SimpleStyleTransfer with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]",
    "MAX_TOTAL_MEMORY" -> (7.5 * CudaMemory.GiB).toString,
    "MAX_DEVICE_MEMORY" -> (7.5 * CudaMemory.GiB).toString,
    "MAX_IO_ELEMENTS" -> (2 * CudaMemory.MiB).toString,
    "CONVOLUTION_WORKSPACE_SIZE_LIMIT" -> (512 * CudaMemory.MiB).toString,
    "MAX_FILTER_ELEMENTS" -> (512 * CudaMemory.MiB).toString
  )

}

object SimpleStyleTransfer_Local extends SimpleStyleTransfer with LocalRunner[Object] with NotebookRunner[Object] {
  override val contentResolution = 300
  override val styleResolution = 400
  override val trainingIterations: Int = 10
  override def inputTimeoutSeconds = 60
}

abstract class SimpleStyleTransfer extends RepeatedArtSetup[Object] {

  val contentUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg"
  val inputUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg"
  val styleUrl = "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg"
  val contentResolution = 512
  val styleResolution = 1280
  val trainingMinutes: Int = 30
  val trainingIterations: Int = 10
  val tileSize = 512
  val tilePadding = 8
  val maxRate = 1e5
  val contentCoeff = 1e-4
  val styleMeanCoeff = 1e0

  override def postConfigure(log: NotebookOutput) = {
    val contentImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(contentUrl, contentResolution)
    }))
    val canvasImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(inputUrl, contentResolution)
    }))
    val styleImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(styleUrl, styleResolution)
    }))
    val contentOperator = new RMSContentMatcher().scale(contentCoeff)
    val styleOperator = new GramMatrixMatcher().combine(new ChannelMeanMatcher().scale(styleMeanCoeff))
    val styleNetwork: PipelineNetwork = log.eval(() => {
      MultiPrecision.setPrecision(SumInputsLayer.combine(
        styleOperator.build(Inc5H_2a, styleImage),
        styleOperator.build(Inc5H_3a, styleImage),
        styleOperator.build(Inc5H_3b, styleImage),
        styleOperator.build(Inc5H_4a, styleImage)
      ), Precision.Float).asInstanceOf[PipelineNetwork]
    })
    withMonitoredImage(log, canvasImage.toRgbImage) {
      withTrainingMonitor(log, trainingMonitor => {
        val trainable = new TiledTrainable(canvasImage, tileSize, tilePadding) {
          override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
            val contentTile = regionSelector.eval(contentImage).getDataAndFree.getAndFree(0)
            regionSelector.freeRef()
            MultiPrecision.setPrecision(SumInputsLayer.combine(
              styleNetwork.addRef(),
              contentOperator.build(Inc5H_3a, contentTile)
            ), Precision.Float).asInstanceOf[PipelineNetwork]
          }
        }
        log.eval(() => {
          val search = new BisectionSearch().setCurrentRate(maxRate / 10).setMaxRate(maxRate).setSpanTol(1e-1)
          new IterativeTrainer(trainable)
            .setOrientation(new TrustRegionStrategy(new LBFGS) {
              override def getRegionPolicy(layer: Layer) = new RangeConstraint().setMin(0e-2).setMax(256)
            })
            .setMonitor(trainingMonitor)
            .setTimeout(trainingMinutes, TimeUnit.MINUTES)
            .setMaxIterations(trainingIterations)
            .setLineSearchFactory((_: CharSequence) => {
              search
            })
            .setTerminateThreshold(java.lang.Double.NEGATIVE_INFINITY)
            .runAndFree
            .asInstanceOf[lang.Double]
        })
      })
    }
  }

}
