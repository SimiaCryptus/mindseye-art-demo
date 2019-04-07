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
import com.simiacryptus.mindseye.art.constraints.{GramMatrixMatcher, RMSChannelEnhancer}
import com.simiacryptus.mindseye.art.models.Inception5H._
import com.simiacryptus.mindseye.lang.cudnn.{CudaMemory, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.SumInputsLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.BisectionSearch
import com.simiacryptus.mindseye.opt.orient.{GradientDescent, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.RangeConstraint
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredImage
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner

object SimpleTexture_EC2 extends SimpleTexture with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 120

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]",
    "MAX_TOTAL_MEMORY" -> (7.5 * CudaMemory.GiB).toString,
    "MAX_DEVICE_MEMORY" -> (7.5 * CudaMemory.GiB).toString,
    "MAX_IO_ELEMENTS" -> (1 * CudaMemory.MiB).toString,
    "CONVOLUTION_WORKSPACE_SIZE_LIMIT" -> (256 * CudaMemory.MiB).toString,
    "MAX_FILTER_ELEMENTS" -> (256 * CudaMemory.MiB).toString
  )

}

object SimpleTexture_Local extends SimpleTexture with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 600

  override val contentResolution = 300
  override val styleResolution = 600
}

abstract class SimpleTexture extends RepeatedArtSetup[Object] {

  val styleUrl = "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg"
  val contentResolution = 512
  val styleResolution = 1280
  val trainingMinutes = 60
  val trainingIterations = 10
  val tileSize = 300

  override def postConfigure(log: NotebookOutput) = {
    val contentImage = Tensor.fromRGB(log.eval(() => {
      new Plasma().paint(contentResolution, contentResolution).toRgbImage
    }))
    val styleImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(styleUrl, styleResolution)
    }))
    val styleNetwork: PipelineNetwork = log.eval(() => {
      val operator = new GramMatrixMatcher()
      MultiPrecision.setPrecision(SumInputsLayer.combine(
        operator.build(Inc5H_2a, styleImage),
        operator.build(Inc5H_3a, styleImage),
        operator.build(Inc5H_3b, styleImage),
        operator.build(Inc5H_4a, styleImage)
      ), Precision.Float).asInstanceOf[PipelineNetwork]
    })
    TestUtil.graph(log, styleNetwork)
    styleNetwork.assertAlive()
    withMonitoredImage(log, contentImage.toRgbImage) {
      withTrainingMonitor(log, trainingMonitor => {
        log.eval(() => {
          val trainable = new TiledTrainable(contentImage, tileSize, 5) {
            override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
              regionSelector.freeRef()
              styleNetwork.addRef()
            }
          }
          val search = new BisectionSearch().setCurrentRate(1e4).setSpanTol(1e-1)
          new IterativeTrainer(trainable)
            .setOrientation(new TrustRegionStrategy(new GradientDescent) {
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

}
